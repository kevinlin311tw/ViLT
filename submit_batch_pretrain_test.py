#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

# ws_config = 'itp_acv' 
ws_config = 'itp_ilp' 
# ws_config = 'itp_east' 
# ws_config = 'vlp_cust' 
# ws_config = 'vlp' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'pretrain'
num_nodes = 2
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes {} --exp_name liqing-vilt-{} " \
             "--config_yaml /raid/keli/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/vilt/{}_nodes{}/pruning_{}_{}_{}/{}_cocotest3'

if num_nodes>1:
    # job_cmd = 'torchlight python run.py with task_mlm_itm whole_word_masking=True  pruning \
    job_cmd = 'lightning python run.py with task_mlm_itm_cocotest whole_word_masking=True  pruning \
        num_gpus={} num_nodes={} per_gpu_batchsize=64 \
        progress_bar_refresh_rate=100 \
        data_root=./pretrain_data/pretrain_arrows  \
        pruning_strategy={} pruning_ratio={} pruning_steps={} seed={}\
        --force \
        '
else:
    job_cmd = 'lightning python run.py with task_mlm_itm_cocotest whole_word_masking=True  pruning \
        num_gpus={} num_nodes={} per_gpu_batchsize=96 \
        progress_bar_refresh_rate=100 \
        data_root=./pretrain_data/pretrain_arrows  \
        pruning_strategy={} pruning_ratio={} pruning_steps={} seed={}\
        --force \
        '

n_repeat = 1
for seed in range(n_repeat):
    for pruning_steps in [100]:
        for pruning_ratio in [0.2]:
            for pruning_strategy in ['small']:
                args = (pruning_strategy, pruning_ratio, pruning_steps, seed)
                resolved_output_dir = output_dir.format(task, num_nodes, *args)
                resolved_job_cmd = job_cmd.format(8, num_nodes, *args)
                resolved_submit_cmd = submit_cmd.format(resolved_output_dir, num_nodes, task, ws_config, resolved_job_cmd)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)
                # exit()
            
