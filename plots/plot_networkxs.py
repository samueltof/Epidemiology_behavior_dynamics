import networkx as nx
import pandas as pd
import os 
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)

figures_path  = config_data.loc['figures_dir'][1]
networks_path = config_data.loc['networks_dir'][1]
num_nodes     = int(config_data.loc['num_nodes'][1])

scale_free5000  = nx.read_gpickle( os.path.join(networks_path, 'scale_free_5000') )
scale_free  = nx.read_gpickle( os.path.join(networks_path, 'scale_free_1000') )
random      = nx.read_gpickle( os.path.join(networks_path, 'random_graph_1000') )
small_world = nx.read_gpickle( os.path.join(networks_path, 'watts_strogatz_1000') )
grid        = nx.grid_graph(dim=[31, 31]).to_undirected()
#grid        = nx.read_gpickle( os.path.join(networks_path, 'grid_1000') )

networks = [scale_free, small_world, random, grid]

### Scale free 5k
plt.figure(figsize=(12,12))
#pos = nx.spring_layout(networks[0], iterations=100) #,prog='twopi',args='')
pos = nx.kamada_kawai_layout(scale_free5000)
#pos = graphviz_layout(scale_free5000, prog='twopi',args='')
nx.draw(G=scale_free5000, pos=pos, 
        node_size=12,
        node_color= 'black',#'#e35340',
        edge_color='gray',
        width=.2,
        edge_cmap=plt.cm.Blues, with_labels=False)
plt.savefig(os.path.join(figures_path, 'scalefree5000.png'), dpi=400, transparent = True, bbox_inches = 'tight', pad_inches = 0.5)



### Scale free
plt.figure(figsize=(12,12))
#pos = nx.spring_layout(networks[0], iterations=100) #,prog='twopi',args='')
pos = nx.kamada_kawai_layout(networks[0])
#pos = graphviz_layout(networks[0], prog='twopi',args='')
nx.draw(G=networks[0], pos=pos, 
        node_size=12,
        node_color= 'black',#'#e35340',
        edge_color='gray',
        width=.2,
        edge_cmap=plt.cm.Blues, with_labels=False)
plt.savefig(os.path.join(figures_path, 'scalefree.png'), dpi=400, transparent = True, bbox_inches = 'tight', pad_inches = 1)


### Small world
plt.figure(figsize=(12,12))
#pos = nx.spring_layout(networks[1], iterations=100) #,prog='twopi',args='')
pos = nx.kamada_kawai_layout(networks[1])
#pos = graphviz_layout(networks[1], prog='twopi', args='')
nx.draw(G=networks[1], pos=pos, 
        node_size=12,
        node_color= 'black',#'#e35340',
        edge_color='gray',
        width=.2,
        edge_cmap=plt.cm.Blues, with_labels=False)
plt.savefig(os.path.join(figures_path, 'small_world.png'), dpi=400, transparent = True, bbox_inches = 'tight', pad_inches = 1)


#### Random
plt.figure(figsize=(12,12))
pos = nx.spring_layout(networks[2], iterations=100) #,prog='twopi',args='')
#pos = graphviz_layout(networks[2], prog='twopi', args='')
#pos = nx.spiral_layout(networks[2])
nx.draw(G=networks[2], pos=pos, 
        node_size=12,
        node_color= 'black',#'#e35340',
        edge_color='gray',
        width=.005,
        edge_cmap=plt.cm.Blues, with_labels=False)
plt.savefig(os.path.join(figures_path, 'random.png'), dpi=400, transparent = True, bbox_inches = 'tight', pad_inches = 1)

#### Grid
plt.figure(figsize=(12,12))
pos = nx.spectral_layout(networks[3])
nx.draw(G=networks[3], pos=pos, 
        node_size=12,
        node_color= 'black', #'#e35340', #'#A0CBE2'
        edge_color='gray',
        width=.2,
        edge_cmap=plt.cm.Blues, with_labels=False)
plt.savefig(os.path.join(figures_path, 'grid_networks_viz.png'), dpi=400, transparent = True, bbox_inches = 'tight', pad_inches = 1)
