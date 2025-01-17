import os
import copy
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
import json
import time
from vilt.utils.pruning import count_flops, prune
import logging



def build_trainer(_config, max_steps, callbacks, logger, grad_steps):
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        progress_bar_refresh_rate=_config["progress_bar_refresh_rate"],
    )
    return trainer

@ex.automain
def main(_config):
    print(_config)
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    # init ViLT and load earlybird weights
    model = ViLTransformerSS(_config)
    # run prunning
    model = prune(_config, model)
    pruned_flops, pruned_num_params = count_flops(model)
    # load pruned weights
    model.load_prune_ckpt()
    # add downstream-specific network
    model.add_downstream_head()

    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    if grad_steps == 0:
        print("batch_size is smaller than num_gpus * per_gpu_batch_size.")
        grad_steps = 1

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    saved_info = {}
    if not _config["test_only"]:
        tic = time.time()
        trainer = build_trainer(_config, max_steps, callbacks, logger, grad_steps)
        trainer.fit(model, datamodule=dm)
        saved_info['train_time'] = round((time.time() - tic) / 3600.0, 2)
    else:
        trainer.test(model, datamodule=dm)

    saved_info['best_score'] = round(model.best_metric*100, 2)
    saved_info['config'] = _config
    json.dump(saved_info, open(os.path.join(_config['log_dir'], 'saved_info.json'), 'w'))