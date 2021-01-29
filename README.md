# How disease risk awareness modulates transmission: coupling infectious disease models with behavioral dynamics

## Dependencies
    python3
    pandas
    numpy
    scipy
    networkx
    python-louvian     # for community detection

## Setup


### Creation of networks using *networkx*

In this investigation, we account for three different types of networks where we ran simulations on: Scale-free, Watts-Strogarz small world and grid graph. To create these networks that are used in simulation, you need to define the number (*n*) of nodes with `--num_nodes <n>`. If not, it is set to *n=1000* automatically. Hence, if you want to create all the networks 

'''
cd models && python create_networkxs.py --num_nodes 1000 --all True && cd - 
'''

    if (isAwesomes){
        return true
    }