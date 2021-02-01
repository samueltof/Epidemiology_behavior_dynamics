# How disease risk awareness modulates transmission: coupling infectious disease models with behavioral dynamics

## Dependencies
    python 3.7
    pandas
    numpy
    scipy
    networkx
    python-louvian 

## Setup
1. Make sure to have dependencies installed
2. Clone our repository `https://github.com/biomac-lab/Epidemiology_behavior_dynamics.git`
3. Create networks
4. Simulate
5. Generate figures

## Usage

### Creation of networks using *networkx*

In this investigation, we account for three different types of networks where we ran simulations on: Scale-free, Watts-Strogarz small world and grid graph. To create these networks that are used in simulation, you need to define the number of nodes (*n*) with `--num_nodes <n>`. If not, it is set to *n=1000* automatically. Hence, if you want to create all the networks specify `--all True`. To do this, run:

    cd models && python create_networkxs.py --num_nodes 1000 --all True && cd - 

Conversely, if you want to create a specific network (say `scale_free`, `small_world` or `grid` ) with a given number of nodes (*n*), set `--specific_network <network_name>`. To create a scale-free network with 5000 nodes, run:

    cd models && python create_networkxs.py --num_nodes 5000 --specific_network scale_free --all False && cd -

### Run simulations

In order to run a simulation, a combination of the infection probability (*beta*) and awareness (*sigma) must be given. This combination are specified on the files `beta_search.csv` and `sigma_search.csv` located on the `run/init_conditions` folder. Each file will contain a row with the parameter value and a key for saving the result (as shown below). We test the model on an interval between 0-1, this means the files `<>_search.csv` contains values in the range of 0 and 1 with space intervals of 0.02 as discribed below:

    key,value
    000,0.00,
    002,0.02,
    .
    .
    098,0.98,
    100,1.00,

The default setting for `sigma_search.csv` are:

    key,value
    100,1.00
    070,0.70
    050,0.50

And for `beta_search.csv` are:

    key,value
    060,0.60
    070,0.70

Besides, the other entries to the model are the `network_type`, the `network_name` (output name after creating the network(s)), the `type_sim` which may be specified as `global` or `local`, the number of iterations `n_iters` (which is 20 by defaul), and the length of simulation given by `max_time` which is set to 150 days. If you want to change the number of iterrations (*iters*), add `--n_iters <iters>` to the command. To change the length of simulation (*days*) add `--max_time <days>`. The execution is then: `cd run && python run_sims.py --network_type <> --network_name <> --type_sim <> && cd -`.

##### For running simulations over a scale-free network with 1000 nodes in both information transmission scheme (as shown in the paper)

    cd run && python run_sims.py --network_type scale_free --network_name scale_free_1000 --type_sim local && cd -
    cd run && python run_sims.py --network_type scale_free --network_name scale_free_1000 --type_sim global && cd -

##### For running simulations over a scale-free network with 5000 nodes in both infomation transmission scheme (as shown in the paper)

    cd run && python run_sims.py --network_type scale_free --network_name scale_free_1000 --type_sim local && cd -
    cd run && python run_sims.py --network_type scale_free --network_name scale_free_1000 --type_sim global && cd -

##### For running simulations over a small-world and grid network with 1000 nodes in both infomation transmission scheme (as shown in the paper)

    cd run && python run_sims.py --network_type small_work --network_name small_work_1000 --type_sim local && cd -
    cd run && python run_sims.py --network_type small_work --network_name small_work_1000 --type_sim global && cd -
    cd run && python run_sims.py --network_type grid --network_name grid_1000 --type_sim local && cd -
    cd run && python run_sims.py --network_type grid --network_name grid_1000 --type_sim global && cd -

##### For running simulations over a ODE (as shown in the paper)


#### Sumulation for cluster analysis

In order to analyse how the community structures (i.e. clusters or hubs) affected infection and behavior, several initial conditions are tested. To generate this initial conditions, run:

    cd run/init_conditions && python create_init_conditios.py && cd -

Each iteration of the simulation records the state of each invididual node and is saved in a `.txt` file.


### Figures generation

#### Networks visualization

In order to visualize the networks you will need to specify the `network_type` (i.e.  `scale_free`, `small_world ` or `grid`) and the `network_name` as it was saven in the `/networks` folder. For visualizing a (already) created scale-free network with 5000 nodes, you would run:

    cd plots && python plot_networks.py --network_type scale_free --network_name scale_free_5000 && cd -

