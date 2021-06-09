#! /home/qing/.virtualenvs/azure/bin/python
import os
import pdb
from datetime import datetime

ws_config = 'itp_acv'
# ws_config = 'vlp_cust' 
# ws_config = 'vlp' 
# ws_config = 'objectdet_wu' 
stamp = ''

if not stamp:
    stamp = datetime.now().strftime('%Y%m%d.%H%M%S')

task = 'vqa'
num_nodes = 1
output_dir = 't-lqing/output/vilt/{}/{}'.format(task, stamp)
submit_cmd = "python -m aml_tools.aml_submit --input_dir . --output_dir {} --num_nodes {} --exp_name liqing-vilt-{} " \
             "--config_yaml ~/.azureml/{}.yaml --cmd \"{}\""

if task == 'caption':
    pass
elif task == 'vqa':
    job_cmd = 'python run.py with task_finetune_vqa_randaug \
    num_gpus=8 num_nodes=1 per_gpu_batchsize=32 \
    progress_bar_refresh_rate=100 \
    data_root=data/VQAv2 load_path="data/vilt_200k_mlm_itm.ckpt" \
    '
elif task == 'pretrain':
    pass

resolved_submit_cmd = submit_cmd.format(output_dir, num_nodes, task, ws_config, job_cmd)
print(resolved_submit_cmd)
os.system(resolved_submit_cmd)
