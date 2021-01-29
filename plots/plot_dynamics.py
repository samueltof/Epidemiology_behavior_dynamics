import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm
import seaborn as sns
import numpy as np
import pandas as pd
import argparse 
import os


parser = argparse.ArgumentParser(description='Time dynamics figures.')

parser.add_argument('--network_type', type=str, default='scale_free',
                    help='Network type for storing...')
parser.add_argument('--network_name', type=str, default='new_scale_free_1000',
                    help='Network type for storing...')
parser.add_argument('--awareness_path', default='param_search/sigma.csv',type=str, 
                    help='Awareness (sigma) for running and saving simulations')
parser.add_argument('--infection_prob_path', default='param_search/beta.csv',type=str, 
                    help='Infection Probability (beta) for running and saving simulations')
parser.add_argument('--type_sim', default='local',type=str, 
                    help='For running local or global simulation')
parser.add_argument('--type_hm', default='R0',type=str, 
                    help='Define the yaxis in heatmaps (R0 or beta)')
parser.add_argument('--type_fig', default='j',type=str, 
                    help='Define heatmaps label display (j of s): j is nice and s is nicer')

args = parser.parse_args()


config_data = pd.read_csv('config.csv', sep=',', header=None, index_col=0)

results_path  = config_data.loc['results_dir'][1]
figures_path  = config_data.loc['figures_dir'][1]
num_nodes =  int(config_data.loc['num_nodes'][1])

df_params = pd.DataFrame(columns=['beta_key', 'sigma_key','beta', 'sigma', 'R0'])

# Selected sigmas and betas
select_sigmas = [0.5,0.7,1.0]  #[0.2, 0.2, 0.7, 1.0]
select_sig_k  = ['050','070','100'] #['020', '020', '070', '100']
select_betas  = [0.6,0.6,0.6] #[0.9,0.9,0.9] #[0.9, 0.9, 0.7, 0.9]
select_bet_k  = ['060','060','060'] #['090','090','090'] #['030', '090', '070', '090']
gamma         = 1/7
repro_numbers = list(np.array(select_betas)/gamma)
num_nodes = 5000

df_params['beta_key']  = select_bet_k
df_params['sigma_key'] = select_sig_k
df_params['beta']      = select_betas
df_params['sigma']     = select_sigmas
df_params['R0']        = repro_numbers

net = 'scale_free'

colors_plt = [ 'tab:red', 'royalblue', 'green'] #, 'tab:purple', 'tab:cyan', 'tab:orange' ]

# Read results
fig, ax = plt.subplots(1,2,figsize=(20, 7))
#fig, ax = plt.subplots(1,2,figsize=(13.2, 5))

for idx, r in tqdm(df_params.iterrows()):
    #path_to_results = os.path.join(results_path, str(num_nodes), args.type_sim, args.network_type, 'dynamics_beta_{}_sigma_{}'.format(r['beta_key'], r['sigma_key']) +'.csv')
    
    #Read global results
    path_to_results_local = os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', 'local', net, 'dynamics_beta_{}_sigma_{}'.format(r['beta_key'], r['sigma_key']) +'.csv')
    res_local = pd.read_csv(path_to_results_local, usecols=['sim_id', 'time', 'S', 'I', 'C','D'])
    res_local_plot = res_local.copy()
    res_local_plot[['S','I','C','D']] = res_local_plot[['S','I','C','D']]/num_nodes
    res_local_plot['type'] = ['global'] * len(res_local_plot)

    #Read local results
    path_to_results_glob = os.path.join(results_path, str(num_nodes)+'_seed_checkpoints_new', 'global', net, 'dynamics_beta_{}_sigma_{}'.format(r['beta_key'], r['sigma_key']) +'.csv')
    res_glob = pd.read_csv(path_to_results_glob, usecols=['sim_id', 'time', 'S', 'I', 'C','D'])
    res_glob_plot = res_glob.copy()
    res_glob_plot[['S','I','C','D']] = res_glob_plot[['S','I','C','D']]/num_nodes
    res_glob_plot['type'] = ['local'] * len(res_glob_plot)

    df_res = [res_local_plot,res_glob_plot]
    df_res_c = pd.concat(df_res)

    # ········· Plot global············

    # Plot disease dynamics

    sns.lineplot( ax = ax[0],
                  data = df_res_c, 
                  x = 'time', y = 'I',
                  label = r'$R_0$={:.1f},$\sigma={:.1f}$'.format(r['R0'],r['sigma']),
                  style='type',
                  color = colors_plt[idx] ,
                  alpha=0.5)
    #ax[0].get_legend().remove()
    ax[0].lines[0].set_linestyle("--")
    ax[0].set_title(r'Disease dynamics $R_0 = {}$'.format(r['R0']),fontsize=22)
    #ax[0].set_xlabel('')
    ax[0].set_xlabel(r'Days',fontsize=21)
    #ax[0].set_xticks(fontsize=20)
    #ax[0].tick_params(axis='x', labelsize=20)
    ax[0].xaxis.set_tick_params(labelsize=20)
    ax[0].yaxis.set_tick_params(labelsize=20)

    ax[0].set_xlim([-0.1,151])
    ax[0].set_ylabel(r'Inf. Fraction ($I$)',fontsize=21)
    #ax[0].set_yticks(fontsize=20)
    ax[0].set_ylim([-0.1,1.1])

    # Plot game dynamics

    sns.lineplot( ax = ax[1],
                  data = df_res_c, 
                  x = 'time', y = 'C',
                  style='type',
                  color = colors_plt[idx],
                  alpha=0.5)
    #ax[1].get_legend().remove()
    ax[1].lines[0].set_linestyle("--")
    ax[1].set_title(r'Behavioral dynamics $R_0 = {}$'.format(r['R0']),fontsize=22)
    ax[1].set_xlabel(r'Days',fontsize=21)
    ax[1].xaxis.set_tick_params(labelsize=20)
    ax[1].yaxis.set_tick_params(labelsize=20)
    #ax[1].set_xticks(fontsize=20)
    ax[1].set_xlim([-0.1,151])
    ax[1].set_ylabel(r'Coop. Fraction ($c$)',fontsize=21)
    #ax[1].set_yticks(fontsize=20)
    ax[1].set_ylim([-0.1,1.1])
    plt.tight_layout()

# plt.savefig(os.path.join(figures_path, 'dynamics', '{}_beta_{}_dynamics.png'.format(net,'0.9')), 
#                             dpi=400, transparent = False, bbox_inches = 'tight', pad_inches = 0.1)
plt.show()


# Figure for legend labels
fig, ax = plt.subplots(2,1,figsize=(9, 8))
for idx, r in tqdm(df_params.iterrows()):
        # Read global results
    path_to_results_local = os.path.join(results_path, str(num_nodes), 'local', 'scale_free', 'dynamics_beta_{}_sigma_{}'.format(r['beta_key'], r['sigma_key']) +'.csv')
    res_local = pd.read_csv(path_to_results_local, usecols=['sim_id', 'time', 'S', 'I', 'C','D'])
    res_local_plot = res_local.copy()
    res_local_plot[['S','I','C','D']] = res_local_plot[['S','I','C','D']]/num_nodes
    res_local_plot['type'] = ['global'] * len(res_local_plot)


    sns.lineplot( ax = ax[0],
                  data = res_local_plot, 
                  x = 'time', y = 'I',
                  label = r'$\sigma={:.1f}$'.format(r['sigma']),
                  color = colors_plt[idx])
    ax[0].get_legend().remove()
    ax[0].set_title(r'Disease dynamics')
    #ax[0].set_xlabel('')
    ax[0].set_xlabel(r'Days',fontsize=21)
    #ax[0].set_xticks(fontsize=20)
    ax[0].set_xticklabels('')
    ax[0].set_xlim([-0.1,151])
    ax[0].set_ylabel(r'Inf. Fraction $\bar{I}$',fontsize=21)
    #ax[0].set_yticks(fontsize=20)
    ax[0].set_ylim([-0.1,1.1])

    plt.figlegend(bbox_to_anchor=(0.7,0.4), fontsize=22)


plt.savefig(os.path.join(figures_path, 'dynamics', 'colorlabel_scale_free_beta_{}_dynamics.png'.format('0.6')), 
                             dpi=400, transparent = False, bbox_inches = 'tight', pad_inches = 0.1)
# plt.show()


# Figure for legend sim types
fig, ax = plt.subplots(2,1,figsize=(9, 8))

# Read global results
path_to_results_local = os.path.join(results_path, str(num_nodes), 'local', 'scale_free', 'dynamics_beta_{}_sigma_{}'.format('060', '100') +'.csv')
res_local = pd.read_csv(path_to_results_local, usecols=['sim_id', 'time', 'S', 'I', 'C','D'])
res_local_plot = res_local.copy()
res_local_plot[['S','I','C','D']] = res_local_plot[['S','I','C','D']]/num_nodes
res_local_plot['type'] = ['Global'] * len(res_local_plot)

# Read local results
path_to_results_glob = os.path.join(results_path, str(num_nodes), 'global', 'scale_free', 'dynamics_beta_{}_sigma_{}'.format('060', '100') +'.csv')
res_glob = pd.read_csv(path_to_results_glob, usecols=['sim_id', 'time', 'S', 'I', 'C','D'])
res_glob_plot = res_glob.copy()
res_glob_plot[['S','I','C','D']] = res_glob_plot[['S','I','C','D']]/num_nodes
res_glob_plot['type'] = ['Local'] * len(res_glob_plot)

df_res = [res_local_plot,res_glob_plot]
df_res_c = pd.concat(df_res)

sns.lineplot( ax = ax[0],
                data = df_res_c, 
                x = 'time', y = 'I',
                style='type', alpha=0.5)
ax[0].get_legend().remove()
ax[0].set_title(r'Disease dynamics')
#ax[0].set_xlabel('')
ax[0].set_xlabel(r'Days',fontsize=21)
#ax[0].set_xticks(fontsize=20)
ax[0].set_xticklabels('')
ax[0].set_xlim([-0.1,151])
ax[0].set_ylabel(r'Inf. Fraction $\bar{I}$',fontsize=21)
#ax[0].set_yticks(fontsize=20)
ax[0].set_ylim([-0.1,1.1])

plt.figlegend(bbox_to_anchor=(0.7,0.4), fontsize=22)

plt.savefig(os.path.join(figures_path, 'dynamics', 'sylelabel_scale_free_beta_{}_dynamics.png'.format('0.6')), 
                             dpi=400, transparent = False, bbox_inches = 'tight', pad_inches = 0.1)

# plt.show()
