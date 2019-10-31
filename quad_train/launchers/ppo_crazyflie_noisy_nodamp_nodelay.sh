#!/bin/bash
parallel ./train_quad.py config/ppo__crazyflie_noisy_nodamp__rew_pos_spin_0.1.yml _results_temp/ppo_crazyflie_noisy_nodamp_nodelay \
--seed {1} \
-p \
env_param.dynamics_change.motor.damp_time_up,\
env_param.dynamics_change.motor.damp_time_down \
-pv {2} \
::: {1..5} ::: 0.0,,0.0