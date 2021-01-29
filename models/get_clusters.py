import networkx as nx
import numpy as np
import pandas as pd
import community as community_louvian
import heapq
from operator import itemgetter
from scipy.stats import mode
from tqdm import tqdm
import os


def get_partition(G, n_biggest=3):

    # part_runs = np.zeros([len(G),runs])
    # for i in range(0,runs):
    #     partition_i    = community_louvian.best_partition(G)     # Get partition/clusters
    #     part_runs[:,i] = np.array(list(partition_i.values()))
    
    # part_mode = mode(part_runs, axis=1)[0].tolist()              # Get mode of partitions
    # part_mode = [int(val[0]) for val in part_mode]
    # nodes_z   = list(np.arange(0,len(G)))
    #partition = dict(zip(nodes_z,part_mode))            # Create dictionary {node:partition}

    partition = community_louvian.best_partition(G, randon_state=0)     # Get partition/clusters

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





#def get_higher_cooperating_clusters(cluster_nodes, disease_checkpoint_path, game_checkpoint_path, iters, beta, sigma):

    # n_clusters = list(cluster_nodes.keys())

    # for it in range(iters+1):

    #     disease_checkpoint = np.loadtxt(os.path.join(disease_checkpoint_path,
    #                          'epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma), delimiter=','))
    #     game_checkpoint    = np.loadtxt(os.path.join(game_checkpoint_path,
    #                          'game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma), delimiter=','))
    
    #     maxtime = disease_checkpoint.shape[0]          # Get time stamp

    #     df_list = []

    #     for cluster, nodes in cluster_nodes.items():  # Calculate dynamics for all clusters

    #         nodes_ = np.array(nodes)

    #         n_def  = np.sum(game_checkpoint[nodes_], axis=0)
    #         n_inf  = np.sum(disease_checkpoint[nodes_], axis=0)
    #         n_tot  = np.array([len(nodes)] * maxtime)
    #         n_cop  = n_tot - n_def
    #         n_sus  = n_tot - n_inf

    #         n_def  = n_def/len(nodes)                  # Normalize
    #         n_inf  = n_inf/len(nodes)
    #         n_cop  = n_cop/len(nodes)
    #         n_sus  = n_sus/len(nodes)

    #         susceptibles = list(n_sus)
    #         infected     = list(n_inf)
    #         defectors    = list(n_def)
    #         cooperators  = list(n_cop)

    #         df_cluster_dynamic = pd.DataFrame(columns=['sim_it','time','cluster','n_nodes','S','I','C','D','beta','sigma','R0']) 

    #         df_cluster_dynamic['sim_it']  = [it] * maxtime
    #         df_cluster_dynamic['cluster'] = cluster
    #         df_cluster_dynamic['n_nodes'] = len(nodes)
    #         df_cluster_dynamic['S']       = susceptibles
    #         df_cluster_dynamic['I']       = infected
    #         df_cluster_dynamic['C']       = cooperators
    #         df_cluster_dynamic['D']       = defectors
    #         df_cluster_dynamic['beta']    = beta
    #         df_cluster_dynamic['sigma']   = sigma
    #         df_cluster_dynamic['R0']      = beta/(1/7)

    #         df_list.append(df_cluster_dynamic)

    # df_return = pd.concat(df_list)

    # df_response_lastweek = df_return.copy()
    # df_response_lastweek = df_response_lastweek.query("time >= 142")
    # # steady state
    # df_response_lastweek = df_response_lastweek.groupby(['cluster','beta', 'sigma', 'R0']).mean()[['S', 'I', 'C','D']].reset_index()





def top_cluster_dynamics(top_cluster_nodes, disease_checkpoint_path, game_checkpoint_path, iters, beta, sigma):

    for it in range(iters+1):

        disease_checkpoint = np.loadtxt(os.path.join(disease_checkpoint_path,
                             'epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma), delimiter=','))
        game_checkpoint    = np.loadtxt(os.path.join(game_checkpoint_path,
                             'game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma), delimiter=','))
    
        maxtime = disease_checkpoint.shape[0]          # Get time stamp

        df_list = []

        for cluster, nodes in top_cluster_nodes.items():

            nodes_ = np.array(nodes)

            n_def  = np.sum(game_checkpoint[nodes_], axis=0)
            n_inf  = np.sum(disease_checkpoint[nodes_], axis=0)
            n_tot  = np.array([len(nodes)] * maxtime)
            n_cop  = n_tot - n_def
            n_sus  = n_tot - n_inf

            n_def  = n_def/len(nodes)                  # Normalize
            n_inf  = n_inf/len(nodes)
            n_cop  = n_cop/len(nodes)
            n_sus  = n_sus/len(nodes)

            susceptibles = list(n_sus)
            infected     = list(n_inf)
            defectors    = list(n_def)
            cooperators  = list(n_cop)

            df_cluster_dynamic = pd.DataFrame(columns=['sim_it','time','cluster','n_nodes','S','I','C','D','beta','sigma','R0']) 

            df_cluster_dynamic['sim_it']  = [it] * maxtime
            df_cluster_dynamic['cluster'] = cluster
            df_cluster_dynamic['n_nodes'] = len(nodes)
            df_cluster_dynamic['S']       = susceptibles
            df_cluster_dynamic['I']       = infected
            df_cluster_dynamic['C']       = cooperators
            df_cluster_dynamic['D']       = defectors
            df_cluster_dynamic['beta']    = beta
            df_cluster_dynamic['sigma']   = sigma
            df_cluster_dynamic['R0']      = beta/(1/7)

            df_list.append(df_cluster_dynamic)

    df_return = pd.concat(df_list)

    return df_return




#game_checkpoint = np.loadtxt('/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000/global/scale_free/checkpoints/epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(itr,beta,sigma), delimiter=',')
#disease_checkpoint = np.loadtxt('/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000/global/scale_free/checkpoints/epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(itr,beta,sigma), delimiter=',')

#maxtime = disease_checkpoint.shape[1]          # Get time stamp

##················· Debug

G = nx.read_gpickle( os.path.join('/Users/samueltorres/Documents/Epidemiology_Replicator/networks/scale_free_5000') )

partition,n_cluster,cluster_nodes,top_clusters,top_cluster_nodes = get_partition(G,3)

beta = 0.6
sigma = 1.0
betak = '060'
sigmak = '100'
itr = 12

df_list_global = []
df_list_local  = []

iters = list(range(itr+1))

from tqdm import tqdm
for itrs in tqdm(iters, total=itr+1):

    # Load checkpoints
    game_checkpoint_glob = np.loadtxt('/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints/global/scale_free/checkpoints/ic_01/game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(itrs,beta,sigma), delimiter=',')
    game_checkpoint_loc  = np.loadtxt('/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints/local/scale_free/checkpoints/ic_01/game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(itrs,beta,sigma), delimiter=',')
    disease_checkpoint_glob = np.loadtxt('/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints/global/scale_free/checkpoints/ic_01/epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(itrs,beta,sigma), delimiter=',')
    disease_checkpoint_loc  = np.loadtxt('/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints/local/scale_free/checkpoints/ic_01/epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(itrs,beta,sigma), delimiter=',')

    maxtime = disease_checkpoint_loc.shape[1]          # Get time stamp

    for cluster, nodes in top_cluster_nodes.items():

        nodes_ = np.array(nodes)

        # Calculate global

        n_def  = np.sum(game_checkpoint_glob[nodes_], axis=0)
        n_inf  = np.sum(disease_checkpoint_glob[nodes_], axis=0)
        n_tot  = np.array([len(nodes)] * maxtime)
        n_cop  = n_tot - n_def
        n_sus  = n_tot - n_inf

        n_def  = n_def/len(nodes)                  # Normalize
        n_inf  = n_inf/len(nodes)
        n_cop  = n_cop/len(nodes)
        n_sus  = n_sus/len(nodes)

        susceptibles = list(n_sus)
        infected     = list(n_inf)
        defectors    = list(n_def)
        cooperators  = list(n_cop)

        df_cluster_dynamic_global = pd.DataFrame(columns=['sim_it','time','cluster','n_nodes','S','I','C','D','beta','sigma','R0']) 

        df_cluster_dynamic_global['sim_it']  = [itrs] * maxtime
        df_cluster_dynamic_global['time']    = list(range(maxtime))
        df_cluster_dynamic_global['cluster'] = cluster
        df_cluster_dynamic_global['n_nodes'] = len(nodes)
        df_cluster_dynamic_global['S']       = susceptibles
        df_cluster_dynamic_global['I']       = infected
        df_cluster_dynamic_global['C']       = cooperators
        df_cluster_dynamic_global['D']       = defectors
        df_cluster_dynamic_global['beta']    = beta
        df_cluster_dynamic_global['sigma']   = sigma
        df_cluster_dynamic_global['R0']      = beta/(1/7)

        df_list_global.append(df_cluster_dynamic_global)

        # Calculate local

        n_def  = np.sum(game_checkpoint_loc[nodes_], axis=0)
        n_inf  = np.sum(disease_checkpoint_loc[nodes_], axis=0)
        n_tot  = np.array([len(nodes)] * maxtime)
        n_cop  = n_tot - n_def
        n_sus  = n_tot - n_inf

        n_def  = n_def/len(nodes)                  # Normalize
        n_inf  = n_inf/len(nodes)
        n_cop  = n_cop/len(nodes)
        n_sus  = n_sus/len(nodes)

        susceptibles = list(n_sus)
        infected     = list(n_inf)
        defectors    = list(n_def)
        cooperators  = list(n_cop)

        df_cluster_dynamic_local = pd.DataFrame(columns=['sim_it','time','cluster','n_nodes','S','I','C','D','beta','sigma','R0']) 

        df_cluster_dynamic_local['sim_it']  = [itrs] * maxtime
        df_cluster_dynamic_local['time']    = list(range(maxtime))
        df_cluster_dynamic_local['cluster'] = cluster
        df_cluster_dynamic_local['n_nodes'] = len(nodes)
        df_cluster_dynamic_local['S']       = susceptibles
        df_cluster_dynamic_local['I']       = infected
        df_cluster_dynamic_local['C']       = cooperators
        df_cluster_dynamic_local['D']       = defectors
        df_cluster_dynamic_local['beta']    = beta
        df_cluster_dynamic_local['sigma']   = sigma
        df_cluster_dynamic_local['R0']      = beta/(1/7)

        df_list_local.append(df_cluster_dynamic_local)

    # df_list_ret.append(df_list)
    # df_return_ = pd.concat(df_list_ret)

df_cluster_dyncs_global_all = pd.concat(df_list_global)
df_cluster_dyncs_local_all = pd.concat(df_list_local)



#················································

##················· Debug

##########################

G = nx.read_gpickle( os.path.join('/Users/samueltorres/Documents/Epidemiology_Replicator/networks/new_scale_free_5000') )

partition,n_cluster,cluster_nodes,top_clusters,top_cluster_nodes = get_partition(G,6)

beta = 0.6
sigma = 0.5
betak = '060'
sigmak = '050'
itr = 12


def cluster_dynamics(top_cluster_nodes, type_sim, results_path, iters, beta, sigma):

    maxtime = 150

    ic_list = []
    
    for ic in tqdm(range(0,10), total=10):

        list_game = []
        list_epid = []


        for it in range(iters+1):

            disease_checkpoint = np.loadtxt(os.path.join(results_path, type_sim, 'scale_free', 'checkpoints', 'ic_0{}'.format(ic),
                                'epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma)), delimiter=',')
            game_checkpoint    = np.loadtxt(os.path.join(results_path, type_sim, 'scale_free', 'checkpoints', 'ic_0{}'.format(ic),
                                'game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma)), delimiter=',')
    
            list_game.append(game_checkpoint)
            list_epid.append(disease_checkpoint)
        
        arr_game = np.dstack(list_game)
        arr_epid = np.dstack(list_epid)

        checks_game_mode = mode(arr_game, axis=2)[0].reshape((5000,150))
        checks_epid_mode = mode(arr_epid, axis=2)[0].reshape((5000,150))

        df_list = []

        for cluster, nodes in top_cluster_nodes.items():

            nodes_ = np.array(nodes)

            n_def  = np.sum(checks_game_mode[nodes_], axis=0)
            n_inf  = np.sum(checks_epid_mode[nodes_], axis=0)
            n_tot  = np.array([len(nodes)] * maxtime)
            n_cop  = n_tot - n_def
            n_sus  = n_tot - n_inf

            n_def  = n_def/len(nodes)                  # Normalize
            n_inf  = n_inf/len(nodes)
            n_cop  = n_cop/len(nodes)
            n_sus  = n_sus/len(nodes)

            susceptibles = list(n_sus)
            infected     = list(n_inf)
            defectors    = list(n_def)
            cooperators  = list(n_cop)

            df_cluster_dynamic = pd.DataFrame(columns=['sim_ic','time','cluster','n_nodes','S','I','C','D','beta','sigma','R0']) 

            df_cluster_dynamic['sim_ic']  = [ic] * maxtime
            df_cluster_dynamic['time']    = list(range(maxtime))
            df_cluster_dynamic['cluster'] = cluster
            df_cluster_dynamic['n_nodes'] = len(nodes)
            df_cluster_dynamic['S']       = susceptibles
            df_cluster_dynamic['I']       = infected
            df_cluster_dynamic['C']       = cooperators
            df_cluster_dynamic['D']       = defectors
            df_cluster_dynamic['beta']    = beta
            df_cluster_dynamic['sigma']   = sigma
            df_cluster_dynamic['R0']      = beta/(1/7)

            df_list.append(df_cluster_dynamic)

        ic_list.append(pd.concat(df_list))


    df_return = pd.concat(ic_list)

    return df_return

res_path = '/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints_new'

df_cluster_dyncs_global = cluster_dynamics(top_cluster_nodes, 'global', res_path, 7, beta, sigma)


df_cluster_dyncs_local = cluster_dynamics(top_cluster_nodes, 'local', res_path, 7, beta, sigma)

import os
os.system('say "your program has finished" ')

#######################

beta = 0.6
sigma = 1.0
betak = '060'
sigmak = '100'

def graph_states(type_sim, results_path, iters, beta, sigma):

    # read both epidemics and game checkpoints
    maxtime = 150
        
    ic = 0

    list_game = []
    list_epid = []

    print('Extracting means')
    for it in range(iters+1):

        disease_checkpoint = np.loadtxt(os.path.join(results_path, type_sim, 'scale_free', 'checkpoints', 'ic_0{}'.format(ic),
                            'epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma)), delimiter=',')
        game_checkpoint    = np.loadtxt(os.path.join(results_path, type_sim, 'scale_free', 'checkpoints', 'ic_0{}'.format(ic),
                            'game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma)), delimiter=',')

        list_game.append(game_checkpoint)
        list_epid.append(disease_checkpoint)
    
    arr_game = np.dstack(list_game)
    arr_epid = np.dstack(list_epid)

    checks_game_mode = mode(arr_game, axis=2)[0].reshape((5000,150))
    checks_epid_mode = mode(arr_epid, axis=2)[0].reshape((5000,150))

    final_checks_game = checks_game_mode[:,149]
    final_checks_epid = checks_epid_mode[:,149]

    states_n = np.zeros_like(final_checks_epid)

    print('Getting new states')
    for i in tqdm(range(0,final_checks_game.shape[0]), total=final_checks_game.shape[0]):

        g_si = final_checks_game[i]
        e_si = final_checks_epid[i]

        # New states 
        # CS: 0, CI:1, DS:2, DI: 3

        if g_si == 0 and e_si == 0:
            states_n[i] = 0

        if g_si == 0 and e_si == 1:
            states_n[i] = 1

        if g_si == 1 and e_si == 0:
            states_n[i] = 2

        if g_si == 1 and e_si == 1:
            states_n[i] = 3

    return states_n

res_path = '/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints_new'

graph_end_states = graph_states( 'global', res_path, 7, beta, sigma)


def game_states(type_sim, results_path, iters, beta, sigma):

    # read both epidemics and game checkpoints
    maxtime = 150
        
    ic = 0

    list_game = []
    list_epid = []

    print('Extracting means')
    for it in range(iters+1):

        game_checkpoint    = np.loadtxt(os.path.join(results_path, type_sim, 'scale_free', 'checkpoints', 'ic_0{}'.format(ic),
                            'game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma)), delimiter=',')

        list_game.append(game_checkpoint)
    
    arr_game = np.dstack(list_game)

    checks_game_mode = mode(arr_game, axis=2)[0].reshape((5000,150))

    final_checks_game = checks_game_mode[:,40]

    return final_checks_game.tolist()

res_path = '/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints_new'

graph_end_states = game_states( 'global', res_path, 7, beta, sigma)

graph_end_states = [int(s) for s in graph_end_states]


#######################

def complete_dynamics(type_sim, results_path, iters, beta, sigma, N=5000):

    maxtime = 150

    ic_list = []~
    
    for ic in tqdm(range(0,10), total=10):

        list_game = []
        list_epid = []


        for it in range(iters+1):

            disease_checkpoint = np.loadtxt(os.path.join(results_path, type_sim, 'scale_free', 'checkpoints', 'ic_0{}'.format(ic),
                                'epid_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma)), delimiter=',')
            game_checkpoint    = np.loadtxt(os.path.join(results_path, type_sim, 'scale_free', 'checkpoints', 'ic_0{}'.format(ic),
                                'game_iter_{}_of_20_beta_{}_sigma_{}.csv'.format(it,beta,sigma)), delimiter=',')
    
            list_game.append(game_checkpoint)
            list_epid.append(disease_checkpoint)
        
        arr_game = np.dstack(list_game)
        arr_epid = np.dstack(list_epid)

        checks_game_mode = mode(arr_game, axis=2)[0].reshape((5000,150))
        checks_epid_mode = mode(arr_epid, axis=2)[0].reshape((5000,150))

        df_list = []

        n_def  = np.sum(checks_game_mode, axis=0)
        n_inf  = np.sum(checks_epid_mode, axis=0)
        n_tot  = np.array([N] * maxtime)
        n_cop  = n_tot - n_def
        n_sus  = n_tot - n_inf

        n_def  = n_def/N                  # Normalize
        n_inf  = n_inf/N
        n_cop  = n_cop/N
        n_sus  = n_sus/N

        susceptibles = list(n_sus)
        infected     = list(n_inf)
        defectors    = list(n_def)
        cooperators  = list(n_cop)

        df_cluster_dynamic = pd.DataFrame(columns=['sim_ic','time','cluster','n_nodes','S','I','C','D','beta','sigma','R0']) 

        df_cluster_dynamic['sim_ic']  = [ic] * maxtime
        df_cluster_dynamic['time']    = list(range(maxtime))
        df_cluster_dynamic['S']       = susceptibles
        df_cluster_dynamic['I']       = infected
        df_cluster_dynamic['C']       = cooperators
        df_cluster_dynamic['D']       = defectors
        df_cluster_dynamic['beta']    = beta
        df_cluster_dynamic['sigma']   = sigma
        df_cluster_dynamic['R0']      = beta/(1/7)

        df_list.append(df_cluster_dynamic)

    ic_list.append(pd.concat(df_list))

    df_return = pd.concat(ic_list)

    return df_return


    res_path = '/Users/samueltorres/Documents/Epidemiology_Replicator/network_results/5000_seed_checkpoints_new'

    res_glob_plot = complete_dynamics('global', res_path, 7, beta, sigma)


