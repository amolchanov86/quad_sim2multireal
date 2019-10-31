#!/usr/bin/env python

import argparse
import json
import os.path as osp
import sys
import time
import numpy as np

import joblib
import tensorflow as tf


def play(pkl_file, n_rollout=20, rollout_length=200):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)
        print("Snapshot content: ", snapshot)
        env = snapshot["env"]
        policy = snapshot["policy"]

        sim_time_vec = []

        # Rollout
        for _ in range(n_rollout):
            sim_time = 0
            obs = env.reset()
            for _ in range(rollout_length):
                sim_time += 1
                env.render()
                action, _ = policy.get_action(obs)
                print("action:", action)
                obs, _, done, _ = env.step(action)
                if done:
                    print("Sim time: ", sim_time)
                    sim_time_vec.append(sim_time)
                    break
        # Report
        print("################################################")
        print("Avg sime time: ", np.mean(sim_time_vec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('pkl_file', metavar='pkl_file', type=str,
                    help='.pkl file containing the policy')
    parser.add_argument('--n_rollout', metavar='n_rollout', type=int,
                    help='Number of rollouts.', default=20)
    parser.add_argument("-l",'--rollout_length', metavar='rollout_length', type=int,
                    help='The length of each rollout.', default=200)
    args = parser.parse_args()

    play(args.pkl_file, n_rollout=args.n_rollout, rollout_length=args.rollout_length)