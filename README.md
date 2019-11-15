# Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors
- Authors: [Artem Molchanov](https://amolchanov86.github.io/), [Tao Chen](https://taochenosu.github.io/), [Wolfgang Hönig](http://act.usc.edu/group.html), [James A. Preiss](http://jpreiss.github.io/), [Nora Ayanian](https://viterbi-web.usc.edu/~ayanian/), [Gaurav S. Sukhatme](http://robotics.usc.edu/~gaurav/)
- Paper Link: [ArXiv](https://arxiv.org/abs/1903.04628)
- Project site: [Google Site](https://sites.google.com/view/sim-to-multi-quad)

<!-- - If you use our work in academic research, please cite us:) -->


## Dependencies

- [Garage](https://github.com/rlworkgroup/garage/)

## Installation

### Step 0

Create directory for all projects:

```sh
mkdir ~/sim2multireal
cd ~/sim2multireal
```

Instead of `~/sim2multireal` you could use any directory you like. It is given just as an example.

### Step 1

Pull [garage](https://github.com/rlworkgroup/garage/).

```sh
git clone https://github.com/rlworkgroup/garage/
```

Checkout the following commit:

```sh
cd garage
git checkout 77714c38d5b575a5cfd6d1e42f0a045eebbe3484
```

Follow the garage [setup instructions](http://rlgarage.readthedocs.io/en/latest/user/installation.html) given below.

The setup requires a MuJoCo key, but since we are not using MuJoCo you can generate a placeholder keyfile.

```sh
touch mjkey.txt
echo "hello" >> mjkey.txt
```

On linux:

```sh
./scripts/setup_linux.sh --mjkey mjkey.txt --modify-bashrc
```

On macOS:

```sh
./scripts/setup_macos.sh --mjkey mjkey.txt --modify-bashrc
```

### Step 2

Clone this repository:

```sh
cd ~/sim2multireal
git clone https://github.com/amolchanov86/quad_sim2multireal.git
cd quad_sim2multireal
```

### Step 3

Install additional dependencies

On linux:

```sh
bash install_depend_linux.sh
```

On macOS:

```sh
bash install_depend_macos.sh
```

### Step 4

Create a new conda environment:

```sh
conda env create -f conda_env.yml
```

## Preparing to run experiements

### General

Each time before running experiments make sure to -

- Activate the conda environment for the experiment
- Add all repos in your `$PYTHONPATH`

```sh
conda activate quad_s2r

export PYTHONPATH=$PYTHONPATH:~/sim2multireal/garage
export PYTHONPATH=$PYTHONPATH:~/sim2multireal/quad_sim2multireal
```

## Experiments

First, go to the root folder:

```sh
cd ~/sim2multireal/quad_sim2multireal/quad_train
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

### Monitoring

#### Use `tensorborad` to monitor the training progress

```sh
tensorboard --logdir ./_results_temp
```

#### To use a specific port

```sh
tensorboard --logdir ./_results_temp --port port_num
```

### Plotting

`plot_tools` library allows nice plotting of statistics.
It assumes that the training results are organized as following: `_results_temp/experiment_folder/seed_{X}/progress.csv` , where:

- `_results_temp`: is the folder containing all experiments and runs.
- `experiment_folder`: is the folder containing an experiment (that could be run with one or multiple seeds).
  They typically named as `param1_paramval1__param2_paramval2`, etc. I.e. they reflect the key parameters and their values in the run.
- `seed_{X}`: is the run folder, i.e. experiment with a particular seed wit value `{X}`

The plot_tools module contains:

- `plot_tools.py`: the library containing all core functionality + it is also a script that can show results of a single experiment. Example:

   ```sh
    ./plot_tools/plot_tools experiment_folder
   ```

- `plot_graphs_with_seeds.py`: a script to plot results with multiple seeds. Example:

   ```sh
   ./plot_tools/plot_graphs_with_seeds.py _results_temp
   ```

Look into `--help` option for all the scripts mentioned above for more options.

### Testing a newly trained model in simulation

`test_controller.py` under `quad_gen` allows you test your fresh model in the simulation with some customizability to the environment. 

Please use `test_controller.py -h` to see the options.

### Generating source code for Crazyflie firmware

`quad_gen` library allow fast generation of embedded source code for the Crazyflie firmware.

Once you have successfully trained a quadrotor stabilizing policy, you will get a pickle file `params.pkl` that is contained in a folder with other data that will be useful for analysis.

In this process, it also assumes the results are organized as following: `_results_temp/experiment_folder/seed_{X}/params.pkl`.

First, go to `~/sim2multireal/quad_sim2multireal/quad_gen`

```sh
cd ~/sim2multireal/quad_sim2multireal/quad_gen
```

#### To generate source code for all training results

```sh
python ./get_models.py 2 _results_temp/ ./models/
```

`_results_temp/` may contain multiple experiments.

#### To generate source code only for the best seeds

```sh
python ./get_models.py 1 _results_temp/ ./models/
```

`_results_temp/` may also contain multiple experiments.

#### To generate source code for selected seeds

```sh
python ./get_models.py 0 _results_temp/ ./models/ -txt [dirs_file]
```

In this case, the `-txt` option is required and allows you to specify relative path (to the `_results_temp/`) of the seeds you would like to generate the source code for. In general when selecting a seed, you will look at the plotting statistics or the tensorboard.
If you use tensorboard, we recommend to look at the position reward and the Gaussian policy variance.

Instead of `./models/` you could use any directory you like. It is given just as an example.
The code for the NN baseline used on the paper is included in `models/` as an example.
