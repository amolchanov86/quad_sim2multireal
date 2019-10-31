# Quadrotor Metalearning Project
## Dependencies:
- [Garage](https://github.com/rlworkgroup/garage/)
- [GymArt](https://github.com/amolchanov86/gym_art.git)

## Installation
### Step 0
Create directory for all projects:
```
mkdir ~/prj
```
Instead of `~/prj` you could use any directory you like. It is given just as an example.

### Step 1
Checkout [garage](https://github.com/rlworkgroup/garage/).

Follow the standard garage [setup instructions](http://rlgarage.readthedocs.io/en/latest/user/installation.html).

Add garage location to your PYTHONPATH. I.e. if garage folder is in ~/prj/garage then
```
export PYTHONPATH=$PYTHONPATH:~/prj
```

### Step 2
Check out this repository:
```sh
cd ~/prj
git clone https://github.com/amolchanov86/quad_dynalearn.git
```

### Step 3 
Install additional dependencies
```sh
bash ~/prj/quad_dynalearn/install.sh
```

### Step 4
Clone GymArt:
```sh
cd ~/prj
git clone https://github.com/amolchanov86/gym_art.git
```

### Step 5
Add all repositories into the `$PYTHONPATH` in your .bashrc:
```sh
export PYTHONPATH=$PYTHONPATH:~/prj/garage
export PYTHONPATH=$PYTHONPATH:~/prj/gym_art
export PYTHONPATH=$PYTHONPATH:~/prj/quad_dynalearn
```


## Preparing to run experiements 

### General
- Activate the anaconda environment for garage
```
conda activate garage
```
- Add all repos in your `$PYTHONPATH` if you haven't done so
- Go to the code root folder:
```
cd ~/prj/quad_dynalearn/quad_dynalearn
```

## Experiments

First, go to the root folder:
```
cd ~/prj/garage_metadist
```

### Training 

#### Train Quadrotor to stabilize at the origin with random horizontal initialization and zero velocity
```sh
./train_garage_quad.py config/ppo_conf.yml _results_temp/ppo_quad_test
```

#### Train Quadrotor to stabilize at the origin with random orientation and random initial velocities
```sh
./train_garage_quad.py config/ppo_conf_randinit.yml _results_temp/ppo_quad_randinit_test
```

#### Train quadrotor MLP dynamics (with Tensorboard reports)
```sh
./train_quad_dynamics_tensorboard.py -o _results_temp/quad_dynamics_stochastic_test
```

#### Train quadrotor stochastic MLP dynamics (with Tensorboard reports)
```sh
./train_quad_dynamics_uncertain.py -o _results_temp/quad_dynamics_stochastic_test
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