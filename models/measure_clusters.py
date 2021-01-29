import networkx as nx
import numpy as np
import pandas as pd
import community as community_louvian
import heapq
from operator import itemgetter
from scipy.stats import mode
from tqdm import tqdm
import os
import powerlaw


def gamma_inference(G):
    # Fit node degree distribution with powerlaw
    # G: undirected scale-free graph
    degree_vals = sorted([d for n,d in G.degree()], reverse=True)
    fit = powerlaw.Fit(degree_vals, xmin=1)
    # fig = fit.plot_pdf(color='b', linewidth=2)
    # fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig)
    gamma = fit.alpha
    return gamma


def get_partition(G, n_biggest=3):

    # part_runs = np.zeros([len(G),runs])
    # for i in range(0,runs):
    #     partition_i    = community_louvian.best_partition(G)     # Get partition/clusters
    #     part_runs[:,i] = np.array(list(partition_i.values()))
    
    # part_mode = mode(part_runs, axis=1)[0].tolist()              # Get mode of partitions
    # part_mode = [int(val[0]) for val in part_mode]
    # nodes_z   = list(np.arange(0,len(G)))
    #partition = dict(zip(nodes_z,part_mode))            # Create dictionary {node:partition}

    partition = community_louvian.best_partition(G)     # Get partition/clusters

    n_cluster = max(partition.values())+1               # Get number of clusters

    cluster_nodes = {}                                  # Get nodes per cluster {cluster:[nodes]}
    
    for key, value in partition.items(): 
        if value not in cluster_nodes: 
            cluster_nodes[value] = [key] 
        else: 
            cluster_nodes[value].append(key)

    n_nodes_in_cluster = { cluster:len(nodes) for cluster, nodes in cluster_nodes.items() }       # Get number of nodes per cluster

    top_clusters = dict(heapq.nlargest(n_biggest, n_nodes_in_cluster.items(), key=itemgetter(1))) # Get n biggest cluster

    # small_clusters = dict(heapq.nsmallest(n_biggest-1, n_nodes_in_cluster.items(), key=itemgetter(1))) # Get n smallest cluster
    
    top_cluster_nodes = {cluster:cluster_nodes[cluster] for cluster,_ in top_clusters.items()}    # Get biggest clusters items

    # small_cluster_nodes = {cluster:cluster_nodes[cluster] for cluster,_ in small_clusters.items()}    # Get smallest clusters items
    
    return partition, n_cluster, cluster_nodes, top_clusters, top_cluster_nodes



G = nx.read_gpickle( os.path.join('/Users/samueltorres/Documents/Epidemiology_Replicator/networks/scale_free_5000') )

partition,n_cluster,cluster_nodes,top_clusters,top_cluster_nodes = get_partition(G,3)



########## Cluster metrics


from networkx.algorithms.community.quality import coverage, modularity, performance
# def get_metrics(G, cluster_nodes):
all_metrics = []
for c, n in top_cluster_nodes.items():                         # Iterate clusters
    comm_i = {node for node in n}                              # Actual community
    G_i    = [node for node in G.nodes()]                      # Complementary nodes
    G_i_c  = [n_ele for n_ele in G_i if n_ele not in comm_i]   # For each cluster, calculate the complementary nodes
    G_i_c  = {node for node in G_i_c}
    part_i = [comm_i,G_i_c]

    df_cluster_metrics = pd.DataFrame(columns=['Cluster','n_nodes','coverage','Q', 'performance'])

    #print('c: ', c,' n: ', len(n),' Q: ', modularity(G,part_i))

    df_cluster_metrics['Cluster']    = [c]
    df_cluster_metrics['n_nodes']    = [len(n)]
    df_cluster_metrics['coverage']   = [coverage(G,part_i)]
    df_cluster_metrics['Q']          = [modularity(G,part_i)]
    df_cluster_metrics['performance'] = [performance(G,part_i)]

    all_metrics.append(df_cluster_metrics)

df_all_metrics = pd.concat(all_metrics)

# Get coverage

