import networkx as nx
import pandas as pd
import os 
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Networks visualizarion')

parser.add_argument('--network_type', type=str,
                    help='Type of network [scale_free,small_world,grid]')
parser.add_argument('--network_name', type=str,
                    help='Name of the network created in the /networks folder')

args = parser.parse_args() 

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)

figures_path  = config_data.loc['figures_dir'][1]
networks_path = config_data.loc['networks_dir'][1]
num_nodes     = int(config_data.loc['num_nodes'][1])

read_network = nx.read_gpickle( os.path.join(networks_path, str(args.network_name)) )

if args.network_type == 'grid':
        plt.figure(figsize=(12,12))
        pos = nx.spectral_layout(args.read_netwrok)
        nx.draw(G=args.read_netwrok, pos=pos, 
                node_size=12,
                node_color= 'black', 
                edge_color='gray',
                width=.2,
                edge_cmap=plt.cm.Blues, with_labels=False)
        plt.savefig(os.path.join(figures_path, 'grid_networks_viz.png'), dpi=400, transparent = True, bbox_inches = 'tight', pad_inches = 1)
        

if (args.network_type == 'scale_free') or (args.network_type == 'small_world'):
        plt.figure(figsize=(12,12))
        pos = nx.kamada_kawai_layout(args.read_network)
        nx.draw(G=args.read_network, pos=pos, 
                node_size=12,
                node_color= 'black',
                edge_color='gray',
                width=.2,
                edge_cmap=plt.cm.Blues, with_labels=False)
        plt.savefig( os.path.join(figures_path, '{}.png'.format(str(args.network_name)) ), dpi=400, transparent = True, bbox_inches = 'tight', pad_inches = 0.5)

else:
        print('Invalid')