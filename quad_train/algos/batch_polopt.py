import time
import h5py
import numbers
import inspect
import os, sys, atexit
import numpy as np

import tensorflow as tf

from garage.algos import RLAlgorithm
import garage.misc.logger as logger
from garage.tf.plotter import Plotter
from garage.tf.samplers import BatchSampler
from garage.tf.samplers import OnPolicyVectorizedSampler

from quad_train.misc.dict2hdf5 import dict2h5 as h5u
from quad_train.misc.video_recorder import VideoRecorder


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def __init__(self,
                 env,
                 policy,
                 baseline,
                 scope=None,
                 n_itr=500,
                 max_samples=None,
                 start_itr=0,
                 batch_size=5000,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 plot=False,
                 pause_for_plot=False,
                 center_adv=True,
                 positive_adv=False,
                 store_paths=False,
                 paths_h5_filename=None,
                 whole_paths=True,
                 fixed_horizon=False,
                 sampler_cls=None,
                 sampler_args=None,
                 force_batch_sampler=False,
                 play_every_itr=None,
                 record_every_itr=None,
                 record_end_ep_num=3,
                 **kwargs):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if
         running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Max umber of iterations.
        :param max_samples: If not None - exit when max env samples is collected (overrides n_itr)
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have
         mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are
         always positive. When used in conjunction with center_adv the
         advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.args = locals()
        del self.args["kwargs"]
        del self.args["self"]
        self.args = {**self.args, **kwargs} #merging dicts

        self.env = env
        try:
            self.env.env.save_dyn_params(filename=logger.get_snapshot_dir().rstrip(os.sep) + os.sep + "dyn_params.yaml")
        except:
            print("WARNING: BatchPolOpt: couldn't save dynamics params")
            # import pdb; pdb.set_trace()
        from gym.wrappers import Monitor
        # self.env_rec = Monitor(self.env.env, logger.get_snapshot_dir() + os.sep + "videos", force=True)

        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.max_samples = max_samples
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.play_every_itr = play_every_itr
        self.record_every_itr = record_every_itr
        self.record_end_ep_num = record_end_ep_num
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = OnPolicyVectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

        ## Initialization of HDF5 logging of trajectories
        if self.store_paths:
            self.h5_prepare_file(filename=paths_h5_filename, args=self.args)
        
        ## Initialize cleaner if we close
        atexit.register(self.clean_at_exit)
    

    def record_policy(self, env, policy, itr, n_rollout=1, path=None, postfix=""):
        # Rollout
        if path is None: path = logger.get_snapshot_dir().rstrip(os.sep) + os.sep + "videos" + os.sep + "itr_%05d%s.mp4" % (itr, postfix)
        path_directory = path.rsplit(os.sep, 1)[0]
        if not os.path.exists(path_directory):
            os.makedirs(path_directory, exist_ok=True)
        for _ in range(n_rollout):
            obs = env.reset()
            recorder = VideoRecorder(env.env, path=path)
            while True:
                # env.render()
                # import pdb; pdb.set_trace()
                action, _ = policy.get_action(obs)
                obs, _, done, _ = env.step(action)
                recorder.capture_frame()
                if done:
                    break
            recorder.close()

    def play_policy(self, env, policy, n_rollout=2):
        # Rollout
        for _ in range(n_rollout):
            obs = env.reset()
            while True:
                env.render()
                action, _ = policy.get_action(obs)
                obs, _, done, _ = env.step(action)
                if done:
                    break

    @staticmethod
    def register_exit_handler(handler_fn):
        # Save will be executed upon normal exit of interpreter
        # NOTE: The functions registered via this module are not called when 
        # the program is killed by a signal not handled by Python
        atexit.register(handler_fn)
    
    def clean_at_exit(self):
        # self.hdf.close()
        pass

    def h5_prepare_file(self, filename, args):
        # Assuming the following structure / indexing of the H5 file
        # teacher_info/
        #   - [teacher_indx]: 
        #        - description
        #        - params
        # traj_data/ 
        #   - [teacher_indx] * [iter_indx] * traj_data

        # Making names and opening h5 file
        if filename is None:
            self.h5_filename = logger.get_snapshot_dir() + os.sep + "trajectories.h5"
        else: #capability to store multiple teachers in a single file
            self.h5_filename = filename
        self.h5_filename = self.h5_filename if self.h5_filename[-3:] == '.h5' else (self.h5_filename + '.h5')

        if os.path.exists(self.h5_filename):
            # input("WARNING: output file %s already exists and will be appended. Press ENTER to continue. (exit with ctrl-C)" % self.h5_filename)
            print("WARNING: output file %s already exists and will be appended" % self.h5_filename)
        self.hdf = h5py.File(self.h5_filename, "a")

        # Creating proper groups
        groups = list(self.hdf.keys())
        # Groups to create: tuples: (group_name, structure_decscripton)
        create_groups = [
            ("teacher_info", "Runs indices(Teachers)"),
            ("traj_data", "Runs(Teachers) x Iterations x Trajectories x Data")
            ]
        
        for group in create_groups:
            if not group in groups:
                self.hdf.create_group(group[0])
                self.hdf[group[0]].attrs["structure"] = np.string_(group[1])
        
        # Checking if other teachers' results already exist in the h5 file
        # If they exist - just append
        teacher_indices = list(self.hdf["traj_data"].keys())
        if not teacher_indices:
            self.teacher_indx = 0
        else:
            teacher_indices = [int(indx) for indx in teacher_indices]
            teacher_indices = np.sort(teacher_indices)
            self.teacher_indx = teacher_indices[-1] + 1
            print("%s : Appended teacher index: " % self.__class__.__name__, self.teacher_indx)
        
        self.hdf.create_group("traj_data/" + h5u.indx2str(self.teacher_indx)) #Teacher group
        
        ## Saving info about the teacher
        teacher_info_group = "teacher_info/" + h5u.indx2str(self.teacher_indx) + "/"
        self.hdf.create_group(teacher_info_group) #Teacher group
        h5u.add_dict(self.hdf, self.args, groupname=teacher_info_group)

        return self.hdf

    def start_worker(self, sess):
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(self.env, self.policy, sess)
            self.plotter.start()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def log_env_info(self, env_infos, prefix=""):
        # Logging rewards
        rew_dic = env_infos["rewards"]
        for key in rew_dic.keys():
            rew_sums = np.sum(rew_dic[key], axis=1)
            logger.record_tabular("rewards/" + key + "_avg", np.mean(rew_sums))
            logger.record_tabular("rewards/" + key + "_std", np.std(rew_sums))


    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
            sess.run(tf.global_variables_initializer())

        # Initialize some missing variables
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                print("Uninitialized var: ", var)
                uninitialized_vars.append(var)
        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        sess.run(init_new_vars_op)

        self.start_worker(sess)
        start_time = time.time()
        last_average_return = None
        samples_total = 0
        for itr in range(self.start_itr, self.n_itr):
            if samples_total >= self.max_samples:
                print("WARNING: Total max num of samples collected: %d >= %d" % (samples_total, self.max_samples))
                break
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                samples_total += self.batch_size
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                last_average_return = samples_data["average_return"]
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                # import pdb; pdb.set_trace()
                if self.store_paths:
                    ## WARN: Beware that data is saved to hdf in float32 by default
                    # see param float_nptype
                    h5u.append_train_iter_data(
                        h5file=self.hdf,  
                        data=samples_data["paths"], 
                        data_group="traj_data/", 
                        teacher_indx=self.teacher_indx, 
                        itr=None,
                        float_nptype=np.float32
                        )
                    # params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                self.log_env_info(samples_data["env_infos"])
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to continue...")
                # Showing policy from time to time
                if self.record_every_itr is not None and self.record_every_itr > 0 and itr % self.record_every_itr == 0:
                    self.record_policy(env=self.env, policy=self.policy, itr=itr)
                if self.play_every_itr is not None and self.play_every_itr > 0 and itr % self.play_every_itr == 0:
                    self.play_policy(env=self.env, policy=self.policy)

        # Recording a few episodes at the end
        if self.record_end_ep_num is not None:
            for i in range(self.record_end_ep_num):
                self.record_policy(env=self.env, policy=self.policy, itr=itr, postfix="_%02d" % i)

        # Reporting termination criteria
        if itr >= self.n_itr-1:
            print("TERM CRITERIA: Max number of iterations reached itr: %d , itr_max: %d" % (itr, self.n_itr-1))
        if samples_total >= self.max_samples:
            print("TERM CRITERIA: Total max num of samples collected: %d >= %d" % (samples_total, self.max_samples))

        self.shutdown_worker()
        if created_session:
            sess.close()

    def log_diagnostics(self, paths):
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

        path_lengths = [path["returns"].size for path in paths]
        logger.record_tabular('ep_len_avg', np.mean(path_lengths))
        logger.record_tabular('ep_len_std', np.std(path_lengths))

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError
