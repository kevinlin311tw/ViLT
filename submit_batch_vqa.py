#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

ws_config = 'itp_acv' 
# ws_config = 'vlp_cust' 
# ws_config = 'vlp' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'vqa'
num_nodes = 1
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes {} --exp_name liqing-vilt-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

output_dir = 't-lqing/experiments/vilt/{}/pruning_{}_{}_{}/{}'
job_cmd = 'python run.py with task_finetune_vqa_randaug pruning \
    num_gpus=8 num_nodes=1 per_gpu_batchsize=32 \
    progress_bar_refresh_rate=100 \
    data_root=data/VQAv2 load_path=data/vilt_200k_mlm_itm.ckpt \
    pruning_strategy={} pruning_ratio={} pruning_steps={} seed={}\
    --force \
    '

n_repeat = 1
for seed in range(n_repeat):
    for pruning_steps in [7000]:
        for pruning_ratio in [0.2, 0.4, 0.6, 0.8]:
            for pruning_strategy in ['small', 'large', 'random']:
                args = (pruning_strategy, pruning_ratio, pruning_steps, seed)
                resolved_output_dir = output_dir.format(task, *args)
                resolved_job_cmd = job_cmd.format(*args)
                resolved_submit_cmd = submit_cmd.format(resolved_output_dir, num_nodes, task, ws_config, resolved_job_cmd)
                print(resolved_submit_cmd)
                os.system(resolved_submit_cmd)
                # exit()
            
