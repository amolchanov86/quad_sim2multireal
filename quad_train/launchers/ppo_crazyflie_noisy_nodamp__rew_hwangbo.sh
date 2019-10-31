#!/bin/bash
parallel ./train_quad.py config/ppo__crazyflie_noisy_nodamp__rew_hwangbo.yml _results_temp/ppo_crazyflie_noisy_nodamp__rew_hwangbo \
--seed {1} \
::: {1..3}