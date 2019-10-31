# Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors
- [Paper](https://arxiv.org/abs/1903.04628)
- [Website](https://sites.google.com/view/sim-to-multi-quad)

## Dependencies:
- [Garage](https://github.com/rlworkgroup/garage/)

## Installation
### Step 0
Create directory for all projects:
```
mkdir ~/prj
```
Instead of `~/prj` you could use any directory you like. It is given just as an example.

### Step 1
Pull [garage](https://github.com/rlworkgroup/garage/).

Checkout the following commit:
```sh
cd ~/prj/garage
git checkout 77714c38d5b575a5cfd6d1e42f0a045eebbe3484
```

Follow the garage [setup instructions](http://rlgarage.readthedocs.io/en/latest/user/installation.html).

### Step 2
Check out this repository:
```sh
cd ~/prj
git clone https://github.com/amolchanov86/quad_sim2multireal.git
```

### Step 3 
Install additional dependencies
```sh
bash ~/prj/quad_sim2multireal/install_depend.sh
```

### Step 4
Add all repositories into the `$PYTHONPATH` (add it to .bashrc):
```sh
export PYTHONPATH=$PYTHONPATH:~/prj/garage
export PYTHONPATH=$PYTHONPATH:~/prj/quad_sim2multireal
```


## Preparing to run experiements 

### General
- Activate the anaconda environment for garage
```
conda activate quad_s2r
```
- Add all repos in your `$PYTHONPATH` if you haven't done so
- Go to the code root folder:
```
cd ~/prj/quad_sim2multireal/quad_train
```

## Experiments

First, go to the root folder:
```
cd ~/prj/quad_sim2multireal/quad_train
```

### Training 

#### Train Quadrotor to stabilize at the origin with random initialization and 5 seeds (you need many seeds since some will fail)
```sh
bash ./launchers/ppo_crazyflie_baseline.sh
```

#### Train Quadrotor to stabilize at the origin with random initialization and a default seed (may fail)
```sh
python ./train_quad.py config/ppo__crazyflie_baseline.yml _results_temp/ppo_crazyflie_baseline/seed_001
```

### Plotting
`plot_tools` library allows nice plotting of statistics.
It assumes that the training results are organized as following: `results_folder/experiment_folder/seed_{X}/progress.csv` , where:
- `results_folder`: is the folder containing all experiments and runs.
- `experiment_folder`: is the folder containing an experiment (that could be run with one or multiple seeds). 
  They typically named as `param1_paramval1__param2_paramval2`, etc. I.e. they reflect the key parameters and their values in the run.
- `seed_{X}`: is the run folder, i.e. experiment with a particular seed wit value `{X}`

The plot_tools module contains:
- `plot_tools.py`: the library containing all core functionality + it is also a script that can show results of a single experiment. Example:
   ```sh
    ./plot_tools/plot_tools experiment_folder
   ```
- `plot_graphs_with_seeds.py `: a script to plot results with multiple seeds. Example:
   ```sh
   ./plot_tools/plot_graphs_with_seeds.py results_folder
   ```

Look into `--help` option for all the scripts mentioned above for more options.