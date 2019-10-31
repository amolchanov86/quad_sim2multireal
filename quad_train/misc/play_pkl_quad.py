#!/usr/bin/env python

import argparse
import json
import os.path as osp
import sys
import time
import numpy as np

import joblib
import tensorflow as tf
import transforms3d as t3d


def R2quat(rot):
    # print('R2quat: ', rot, type(rot))
    R = rot.reshape([3,3])
    w = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    w4 = (4.0 * w)
    x = (R[2,1] - R[1,2]) / w4
    y = (R[0,2] - R[2,0]) / w4
    z = (R[1,0] - R[0,1]) / w4
    return np.array([w,x,y,z])

def play(pkl_file, n_rollout=20, rollout_length=200, lowpass_coeff=0.9,
        obs_params=None, csv_filename=None, scale_down_inertia=1.0):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)
        env = snapshot["env"]
        policy = snapshot["policy"]

        sim_time_vec = []

        # Observations
        obs_vec = []
        action_vec = []

        recording_time = 4.0
        inertia_scale = 1.0

        # Rollout
        for n in range(n_rollout):
            csv_data = []
            sim_time = 0
            time = 0

            print("Scale:", 1.0/inertia_scale, "Inertia: ", env.env.dynamics.inertia)
            print("T2W:", env.env.dynamics.thrust_to_weight, "goal:", env.env.goal, "sample_goal:", env.env.resample_goal)
            policy.reset()
            obs = env.reset()
            for t in range(rollout_length):
                # print("obs:", obs, "shape: ", obs.shape)
                # print("obs:", obs[:3])
                sim_time += 1
                time += 1./env.env.control_freq
                env.render()
                # obs = np.append(obs, 2. + obs[2])
                obs_vec.append(obs)
                action, action_components = policy.get_action(obs)
                action_vec.append(action_components["mean"])
                if t == 0:
                    action_lowpass = action
                else:
                    action_lowpass = (1-lowpass_coeff) * action + lowpass_coeff * action_lowpass
                # print("action:", action_lowpass)

                # quat = R2quat(rot=obs[6:15])
                if time > recording_time:
                    quat = t3d.quaternions.mat2quat(obs[6:15])
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
                    csv_data.append(np.concatenate([np.array([1.0/env.env.control_freq * t]), obs[0:3], quat]))

                obs, _, done, _ = env.step(action_components["mean"])

                if done:
                    print("Sim time: ", sim_time)
                    sim_time_vec.append(sim_time)
                    break
            
            if csv_filename is not None:
                import csv
                with open(csv_filename + "_I_downscale_%.2f.csv" % (1.0/inertia_scale), mode="w") as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',')
                    csv_writer.writerow(["time","x","y","z", "qx","qy","qz","qw"])
                    for row in csv_data:
                        csv_writer.writerow([i for i in row])
            # if n == 0:
            #     env.env.dynamics.inertia = env.env.dynamics.inertia / 2.0
            #     inertia_scale = 0.5
            # else:
            env.env.dynamics.inertia = env.env.dynamics.inertia * scale_down_inertia
            inertia_scale *= scale_down_inertia

        # Report
        print("################################################")
        print("Avg sime time: ", np.mean(sim_time_vec))

        import matplotlib.pyplot as plt

        # import pdb; pdb.set_trace()
        if obs_params is not None:
            obs_sizes = [obs_params[key] for key in obs_params.keys()]
            obs_indices = np.cumsum(obs_sizes)
            obs_vec = np.split(np.array(obs_vec).T, obs_indices, axis=0)
            obs_ids_total = len(obs_params)
            obs_num = 0
            for obs_id in obs_params.keys():
                plt.subplot(obs_ids_total + 1, 1, obs_num+1)
                for obs_i in range(obs_vec[obs_num].shape[0]):
                    plt.plot(obs_vec[obs_num][obs_i, :])
                plt.title(obs_id)
                obs_num += 1

            ## Actions
            plt.subplot(obs_ids_total + 1, 1, obs_num+1)
            action_vec = np.array(action_vec).T
            for act in range(action_vec.shape[0]):
                plt.plot(action_vec[act])
            plt.title("Actions")
        else:    
            ## Plotting actions
            plt.figure(2)
            action_vec = np.array(action_vec).T
            for act in range(action_vec.shape[0]):
                plt.plot(action_vec[act])
            plt.title("Actions")
            plt.legend(list(range(action_vec.shape[0])))
        
        plt.show(block=False)
        input("Press ENter to continue ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('pkl_file',
                type=str,
                help='.pkl file containing the policy')
    parser.add_argument("-n",'--n_rollout',
                type=int,
                help='Number of rollouts.', default=20)
    parser.add_argument("-l",'--rollout_length', 
                type=int,
                help='The length of each rollout.', default=400)
    parser.add_argument("-obs",'--obs_params', 
                help='Observation components: name1,size1,,name2,size2  etc.', 
                default="xyz,3,,Vxyz,3,,R,9,,Omega,3,,Z,1")
    parser.add_argument("-pltobs",'--plot_obs',
                help='Plot observations', 
                action="store_true")
    parser.add_argument("-lpc",'--lowpass_coeff', 
            help='Low pass coefficient for printing data',
            type=float,
            default=0.9)
    parser.add_argument("-is",'--inertia_scale', 
            help='How much to scale down inertia at each rollout',
            type=float,
            default=1.0)
    parser.add_argument(
        '-csv',"--csv_filename",
        help="Filename-base for qudrotor data"
    )
    args = parser.parse_args()

    obs_params = [obs.split(",") for obs in args.obs_params.split(",,")]
    obs_params = dict([(obs[0], int(obs[1])) for obs in obs_params])
    if not args.plot_obs:
        obs_params = None

    play(args.pkl_file, 
        n_rollout=args.n_rollout, 
        rollout_length=args.rollout_length, 
        obs_params=obs_params,
        lowpass_coeff=args.lowpass_coeff,
        csv_filename=args.csv_filename,
        scale_down_inertia=args.inertia_scale
        )