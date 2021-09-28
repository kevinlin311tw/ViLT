# python run.py with data_root=data/VQAv2 num_gpus=1 num_nodes=1 per_gpu_batchsize=64 task_finetune_vqa_randaug test_only=True load_path="data/vilt_vqa.ckpt"
# ln -s /raid/keli/azure_storage/vig_data/jianfw/data/ziyi_arrow/data pretrain_data

CUDA_VISIBLEi_DEVICES='7' python run.py with task_mlm_itm_cocotest pruning \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=32 \
    progress_bar_refresh_rate=10 \
    pruning_ratio=0.2 pruning_steps=100 \
    data_root=pretrain_data/pretrain_arrows  \
    --force \
    |& tee output/log.txt
