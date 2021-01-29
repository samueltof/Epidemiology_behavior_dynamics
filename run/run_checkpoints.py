import numpy as np
import pandas as pd
import os
import random 
import networkx as nx
from models import sis_replicator

import argparse 

parser = argparse.ArgumentParser(description='Network simulations.')

parser.add_argument('--network_type', type=str, default='scale_free',
                    help='Network type for storing...')
parser.add_argument('--network_name', type=str, default='new_scale_free_5000',
                    help='Network type for storing...')
parser.add_argument('--type_sim', default='global',type=str, 
                    help='For running local or global simulation')

args = parser.parse_args()

# sigma_search = pd.read_csv(args.awareness_path, dtype={'key':str, 'value':float})
# beta_search  = pd.read_csv(args.infection_prob_path, dtype={'key':str, 'value':float})


config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)

networks_path = config_data.loc['networks_dir'][1]
results_path  = config_data.loc['results_dir'][1]
num_nodes     = int(config_data.loc['num_nodes'][1])
if args.network_name == 'new_scale_free_5000':
    num_nodes = 5000

G = nx.read_gpickle( os.path.join(networks_path, args.network_name) )


df_params = pd.DataFrame(columns=['beta_key', 'sigma_key','beta', 'sigma', 'R0'])

# Selected sigmas and betas
select_sigmas = [1.0,0.7,0.5] #[0.7]  #[0.2, 0.7, 1.0, 1.0]
select_sig_k  = ['100','070','050'] #['070']  #['020', '070', '100', '100']
select_betas  = [0.6,0.6,0.6] #[0.7] #[0.3, 0.6, 0.6, 0.9]
select_bet_k  = ['060','060','060'] #['070'] #['030', '060', '060', °'090']
gamma         = 1/7
repro_numbers = list(np.array(select_betas)/gamma)

df_params['beta_key']  = select_bet_k
df_params['sigma_key'] = select_sig_k
df_params['beta']      = select_betas
df_params['sigma']     = select_sigmas
df_params['R0']        = repro_numbers


from models import run_model 

# df = pd.concat([sigma_search, beta_search], axis=1)

# df_param_run = pd.DataFrame(columns=['beta_key', 'sigma_key', 'beta_val', 'sigma_val'])

# beta_key  = []
# sigma_key = []
# beta_val  = []
# sigma_val = []
# for idx_sigma , r_sigma in sigma_search.iterrows():
#     for idx_beta , r_beta in beta_search.iterrows():

#         beta_key.append( r_beta['key']   )
#         sigma_key.append( r_sigma['key'] )
#         beta_val.append( r_beta['value'] )
#         sigma_val.append( r_sigma['value'] )

# df_param_run['beta_key'] = beta_key  
# df_param_run['sigma_key'] = sigma_key 
# df_param_run['beta_val'] = beta_val  
# df_param_run['sigma_val'] = sigma_val 


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
os.system('say "your program has finished" ')