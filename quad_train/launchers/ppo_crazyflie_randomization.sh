#!/bin/bash
parallel ./train_quad.py config/ppo__crazyflie_randomization_0.2_noisy_nodamp.yml _results_temp/ppo_crazyflie_randomization \
-c \
--seed {1} \
-p env_param.dynamics_randomization_ratio \
-pv {2} \
::: {1..5} ::: 0.1 0.2 0.3