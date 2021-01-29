python run_checkpoints.py --network_type scale_free --network_name scale_free_1000 --type_sim global 
python run_checkpoints.py --network_type scale_free --network_name scale_free_5000 --type_sim global
python run_checkpoints.py --network_type scale_free --network_name new_scale_free_5000 --type_sim global 
python run_checkpoints.py --network_type watts_strogatz --network_name watts_strogatz_1000 --type_sim global 
python run_checkpoints.py --network_type grid --network_name grid_1000 --type_sim global 

python run_checkpoints.py --network_type scale_free --network_name scale_free_1000 --type_sim local 
python run_checkpoints.py --network_type scale_free --network_name scale_free_5000 --type_sim local 
python run_checkpoints.py --network_type scale_free --network_name new_scale_free_5000 --type_sim local 
python run_checkpoints.py --network_type grid --network_name grid_1000 --type_sim local
python run_checkpoints.py --network_type grid --network_name grid_1000 --type_sim local 