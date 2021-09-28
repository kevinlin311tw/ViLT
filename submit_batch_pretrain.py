#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

ws_config = 'itp_acv' 
# ws_config = 'itp_east' 
# ws_config = 'itp_ilp' 
# ws_config = 'vlp_cust' 
# ws_config = 'vlp' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'pretrain'
num_nodes = 4
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes {} --exp_name liqing-vilt-{} " \
             "--config_yaml /raid/keli/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/vilt/{}_nodes{}_torch18/pruning_{}_{}_{}/{}'

job_cmd = 'lightning python run.py with task_mlm_itm_pretrain_only step200k whole_word_masking=True pruning \
    num_gpus={} num_nodes={} per_gpu_batchsize=96 \
    progress_bar_refresh_rate=100 \
    data_root=./pretrain_data/pretrain_arrows  \
    pruning_strategy={} pruning_ratio={} pruning_steps={} seed={}\
    --force \
    '

n_repeat = 1
for seed in range(n_repeat):
    for pruning_steps in [10000]: # [1000, 2000, 5000, 10000]:
        for pruning_ratio in [0.8]: #  [0.2, 0.4, 0.6, 0.8]:
            for pruning_strategy in ['small']:
                args = (pruning_strategy, pruning_ratio, pruning_steps, seed)
                resolved_output_dir = output_dir.format(task, num_nodes, *args)
                resolved_job_cmd = job_cmd.format(8, num_nodes, *args)
                resolved_submit_cmd = submit_cmd.format(resolved_output_dir, num_nodes, task, ws_config, resolved_job_cmd)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)
                # exit()
            
