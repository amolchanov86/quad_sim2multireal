#!/usr/bin/env python

import argparse
import os.path as osp
import sys
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle

from gym_art.quadrotor.quadrotor import QuadrotorEnv


def play(quad_params, u_freqs, u_magns, obs_params, out_filename=None, render=True, plot=True, gravity=0., ep_time=1., conf_name=""):
    # Modifying some parameters
    quad_params["gravity"] = gravity
    quad_params["init_random_state"] = False
    quad_params["dynamics_change"]["noise"]["thrust_noise_ratio"] = 0.

    quad_params["ep_time"] = ep_time
    env = QuadrotorEnv(**quad_params)
    control_freq = env.control_freq
    ep_len = env.ep_len
    u_dt = 1./control_freq
    dt = env.dt
    
    obs_descr_str = "freq x magn x time x obs"
    act_descr_str = "freq x magn x time x actions"

    # import pdb;pdb.set_trace()

    obs_params = [obs.split(",") for obs in args.obs_params.split(",,")]
    obs_components = [(obs[0], int(obs[1])) for obs in obs_params]
    obs_params = dict(obs_components)

    sim_time_vec = []

    # Observations
    obs_vec = np.zeros([len(u_freqs), len(u_magns)] + [ep_len + 1] + list(env.observation_space.high.shape))
    action_vec = np.zeros([len(u_freqs), len(u_magns)] + [ep_len + 1] + list(env.action_space.high.shape))

    # Action generator (freq == Hz)
    def get_action(t, magn, freq, offset=0.0):
        u01 = magn * np.sin(2 * np.pi * freq * t) + offset
        u23 = magn * np.sin(2 * np.pi * freq * t + np.pi) + offset
        return np.array([u01,u01,u23,u23])

    for u_f_i, u_f in enumerate(u_freqs):
        for u_m_i, u_m in enumerate(u_magns):
            sim_step = 0
            obs = env.reset()
            while True:
                if render: env.render()
                obs_vec[u_f_i, u_m_i, sim_step, ...] = obs
                action = get_action(u_dt * sim_step, magn=u_m, freq=u_f)
                action_vec[u_f_i, u_m_i, sim_step, ...] = action
                obs, _, done, _ = env.step(action)
                
                if done:
                    print("Sim time: ", u_dt * sim_step)
                    sim_time_vec.append(u_dt * sim_step)
                    break
                sim_step += 1

    # Report
    print("################################################")
    print("Avg sim time: ", np.mean(sim_time_vec))

    ## Pickle everything
    if out_filename is not None:
        data = {"obs": obs_vec, "actions": action_vec, 
            "time": np.linspace(0, env.ep_time, ep_len),
            "description": 
                {"obs_mx_dims": obs_descr_str,
                 "obs_components": obs_components,
                 "actions_mx_dims": act_descr_str,
                 "actions_freq": u_freqs,
                 "actions_magn": u_magns
                 } 
            }
        with open(out_filename, 'wb') as handle:
            pickle.dump(data, handle)

    # import pdb; pdb.set_trace()
    if plot:
        obs_sizes = [obs_params[key] for key in obs_params.keys()]
        obs_indices = np.cumsum(obs_sizes)
        obs_vec = np.split(obs_vec, obs_indices[:-1], axis=3)
        obs_num = 0
        for obs_num, obs_id in enumerate(obs_params.keys()):
            plt.figure(obs_num+1)
            for obs_i in range(obs_sizes[obs_num]):
                ax = plt.subplot(obs_sizes[obs_num] + 2, 1, obs_i+1)
                
                for f_i, u_f in enumerate(u_freqs):
                    for m_i, u_m in enumerate(u_magns):
                        plt.plot(obs_vec[obs_num][f_i, m_i, :, obs_i], label=conf_name + ": freq: %.2f magn: %.2f" % (u_f, u_m))
                    plt.legend(loc="right", borderaxespad=0.)

                
                ax.text(.5,.9, obs_id + "_" + str(obs_i),
                    horizontalalignment='center',
                    transform=ax.transAxes)
            
            ## Actions (on each graph for convenience)
            ax = plt.subplot(obs_sizes[obs_num] + 2, 1, obs_i+2)
            for f_i, u_f in enumerate(u_freqs):
                for m_i, u_m in enumerate(u_magns):
                    for act in range(2):
                        plt.plot(action_vec[f_i, m_i, :, act], label=str(act))
                    ax.text(.5,.9, "Actions",
                        horizontalalignment='center',
                        transform=ax.transAxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument("-c","--config",
                help='Config file for a quadrotor env or a list of such files', nargs='+')    
    parser.add_argument("-obs",'--obs_params', 
                help='Observation components: name1,size1,,name2,size2  etc.', 
                default="xyz,3,,Vxyz,3,,R,9,,Omega,3")
    parser.add_argument("-p",'--plot',
                help="Set it if you dont want to plot data", 
                action="store_false")
    parser.add_argument("-f",'--freq',
                help="List of frequencies to apply (Hz) separated by comma", 
                default="1,10")
    parser.add_argument("-m",'--magn',
            help="List of magnitudes to apply separated by comma", 
            default="0.1,0.5") 
    parser.add_argument("-o",'--out',
        help="Output pickle filename") 
    parser.add_argument("-r",'--render',
        help="Set this flag if you don't want to render",
        action="store_false") 
    parser.add_argument("-g",'--gravity',
        help="Gravity",
        type=float,
        default=0.) 
    parser.add_argument("-t",'--ep_time',
        help="Time (s) for the episode",
        type=float,
        default=3.) 
    args = parser.parse_args()

    freq = [float(freq) for freq in args.freq.split(",")]
    magn = [float(magn) for magn in args.magn.split(",")]

    # import pdb; pdb.set_trace()

    quad_params = []
    for confname in args.config:
        import yaml
        yaml_stream = open(confname, 'r')
        config = yaml.load(yaml_stream)

        if "variant" in config:
            quad_params.append(config["variant"]["env_param"])
        else:
            quad_params.append(config)

    for quad_id, quad in enumerate(quad_params):
        print("####################################################")
        print("Running: ", args.config[quad_id])
        play(quad_params=quad, 
            obs_params=args.obs_params,
            u_freqs=freq, 
            u_magns=magn, 
            out_filename=args.out, 
            render=args.render, 
            plot=args.plot,
            gravity=args.gravity,
            ep_time=args.ep_time,
            conf_name="conf_" + str(quad_id)
            )
    
    plt.show(block=False)
    input("Press ENter to continue ...")

## Unpickle example
# file = open("_results_temp/yaw0/quad_oscilations_sim100hz.pkl", "rb")                                                                              
# data = pickle.load(file) 