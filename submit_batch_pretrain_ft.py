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

task = 'vqa'
num_nodes = 1
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes {} --exp_name liqing-vilt-pretrain-ft-{} " \
             "--config_yaml /raid/keli/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/vilt/pretrain_nodes4_torch18/pruning_{}_{}_{}/{}/{}'

job_cmd = 'lightning python run_prune_ft.py with task_finetune_vqa_randaug pruning \
    num_gpus=8 num_nodes=1 per_gpu_batchsize=32 \
    progress_bar_refresh_rate=100 \
    data_root=data/VQAv2 \
    earlybird_path=experiments/vilt/pretrain_nodes4_torch18/pruning_{}_{}_{}/0/mlm_itm_seed0_from_/version_0/checkpoints/epoch=6-step=19999.ckpt \
    pruned_path=experiments/vilt/pretrain_nodes4_torch18/pruning_{}_{}_{}/0/mlm_itm_seed0_from_/version_0/checkpoints/last.ckpt \
    pruning_strategy=small pruning_ratio={} seed=0\
    --force \
    '

n_repeat = 1
for seed in range(n_repeat):
    for pruning_steps in [20000]: # [1000, 2000, 5000, 10000]:
        for pruning_ratio in [0.4, 0.6, 0.8]: #  [0.2, 0.4, 0.6, 0.8]:
            for pruning_strategy in ['small']:
                args = (pruning_strategy, pruning_ratio, pruning_steps)
                resolved_output_dir = output_dir.format(*args, task, seed)
                resolved_job_cmd = job_cmd.format(*args, *args, pruning_ratio)
                resolved_submit_cmd = submit_cmd.format(resolved_output_dir, num_nodes, task, ws_config, resolved_job_cmd)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)
                # exit()
            
