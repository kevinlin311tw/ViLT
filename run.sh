# python run.py with data_root=data/VQAv2 num_gpus=1 num_nodes=1 per_gpu_batchsize=64 task_finetune_vqa_randaug test_only=True load_path="data/vilt_vqa.ckpt"
python run.py with task_finetune_vqa_randaug \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=32 \
    progress_bar_refresh_rate=10 \
    data_root=data/VQAv2 load_path="data/vilt_200k_mlm_itm.ckpt"