import numpy as np
import pandas as pd
import os
import sys
import networkx as nx
from models import sis_replicator

import argparse 

parser = argparse.ArgumentParser(description='Network simulations.')

parser.add_argument('--network_type', type=str, default='scale_free',
                    help='Network type for storing...')
parser.add_argument('--network_name', type=str, default='new_scale_free_5000',
                    help='Network type for storing...')
parser.add_argument('--beta', type=float,
                    help='Specify the infection probability')
parser.add_argument('--sigma', type=float,
                    help='Specify the awareness')
parser.add_argument('--type_sim', default='global',type=str, 
                    help='For running local or global simulation')

args = parser.parse_args()

config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)

networks_path = config_data.loc['networks_dir'][1]
results_path  = config_data.loc['results_dir'][1]
num_nodes     = int(config_data.loc['num_nodes'][1])
if args.network_name == 'new_scale_free_5000':
    num_nodes = 5000

G = nx.read_gpickle( os.path.join(networks_path, args.network_name) )


df_params = pd.DataFrame(columns=['beta', 'sigma', 'R0'])

# Selected sigmas and betas
select_sigmas = [args.sigma]
select_betas  = [args.beta]
gamma         = 1/7
repro_numbers = list(np.array(select_betas)/gamma)

df_params['beta']      = select_betas
df_params['sigma']     = select_sigmas
df_params['R0']        = repro_numbers

sys.path.append('../')
from models import models 


initConditions = pd.read_csv('init_conditions/initial_conditions.csv')



if args.type_sim=='local':
    local = False
elif args.type_sim=='global':
    local = True

print('Running simulations for {} network in {}  scheme\n'.format(args.network_type, args.type_sim))

from tqdm import tqdm

for i in tqdm(range(0,len(initConditions.index)), total = len(initConditions.index)):
    print('Solving for '+str(i)+' initial params')

    for idx, r in df_params.iterrows():

        # Get initial parameters
        initInf_i = initConditions['I'][i]
        initDef_i = np.fromstring(initConditions['D'][i], sep = '|').astype(int)

        # Parameters
        model_params = {}
        model_params['time2Recover']  = 7
        model_params['probInfect']    = r['beta']
        model_params['awareness']     = r['sigma']
        model_params['initInfected']  = initInf_i
        model_params['initDefectors'] = initDef_i

        if not os.path.isdir( os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', args.type_sim, args.network_type) ):
            os.makedirs(os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', args.type_sim, args.network_type))
            
        if not os.path.isdir( os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', args.type_sim, args.network_type, 'checkpoints', 'ic_0{}'.format(i)) ):
            os.makedirs(os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', args.type_sim, args.network_type, 'checkpoints', 'ic_0{}'.format(i)))

        path_to_save_checkpoints = os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', args.type_sim, args.network_type, 'checkpoints', 'ic_0{}'.format(i))
        path_to_save_response    = os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', args.type_sim, args.network_type, 'dynamics_beta_{}_sigma_{}'.format(r['beta_key'], r['sigma_key']) +'.csv')

        # if os.path.exists(path_to_save_response):
        #     continue
        #print( 'Running for beta={}, sigma={} \r'.format(r['beta'], r['sigma']) )

        df_response = run_model(sis_replicator, G , params=model_params, n_iters=20, max_time=150, num_checkpoints=8, local=local, path_to_save_checkpoints= path_to_save_checkpoints)
        df_response.to_csv( path_to_save_response )

print('\t DONE!\n')

import os
os.system('say "your program has finished"to_undirected