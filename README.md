# How disease risk awareness modulates transmission: coupling infectious disease models with behavioral dynamics

## Dependencies
    python3
    pandas
    numpy
    scipy
    networkx
    python-louvian 

## Setup
1. Make sure to have dependencies installes
2. Clone our repository `https://github.com/biomac-lab/Epidemiology_behavior_dynamics.git`
3. Create networks
4. Simulate

## Usage

### Creation of networks using *networkx*

In this investigation, we account for three different types of networks where we ran simulations on: Scale-free, Watts-Strogarz small world and grid graph. To create these networks that are used in simulation, you need to define the number of nodes (*n*) with `--num_nodes <n>`. If not, it is set to *n=1000* automatically. Hence, if you want to create all the networks specify `--all True`. To do this, run:

    cd models && python create_networkxs.py --num_nodes 1000 --all True && cd - 

Conversely, if you want to create a specific network (say `scale_free`, `small_worl` or `grid` ) with a given number of nodes (*n*), set `--specific_network <network_name>`. To create a scale-free network, run:

    cd models && python create_networkxs.py --num_nodes 5000 --specific_network scale_free --all False && cd -

