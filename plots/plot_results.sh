## R0 plots
# Global results
python plot_results.py --network_type scale_free --type_sim global --type_fig j --type_hm R0
#python plot_results.py --network_type random --type_sim global --type_fig j --type_hm R0
python plot_results.py --network_type watts_strogatz --type_sim global --type_fig j --type_hm R0
python plot_results.py --network_type grid --type_sim global --type_fig j --type_hm R0
# Local results
python plot_results.py --network_type scale_free --type_sim local --type_fig j --type_hm R0
#python plot_results.py --network_type random --type_sim local --type_fig j --type_hm R0
python plot_results.py --network_type watts_strogatz --type_sim local --type_fig j --type_hm R0
python plot_results.py --network_type grid --type_sim local --type_fig j --type_hm R0

## beta plots
# Global results
python plot_results.py --network_type scale_free --type_sim global --type_fig j --type_hm beta
#python plot_results.py --network_type random --type_sim global --type_fig j --type_hm beta
python plot_results.py --network_type watts_strogatz --type_sim --type_fig j global --type_hm beta
python plot_results.py --network_type grid --type_sim global --type_fig j --type_hm beta
# Local results
python plot_results.py --network_type scale_free --type_sim local --type_fig j --type_hm beta
#python plot_results.py --network_type random --type_sim local --type_fig j --type_hm beta
python plot_results.py --network_type watts_strogatz --type_sim local --type_fig j --type_hm beta
python plot_results.py --network_type grid --type_sim local --type_fig j --type_hm beta