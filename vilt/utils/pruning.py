from thop import profile
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from vilt.modules.vision_transformer import Mlp, Attention
from easydict import EasyDict
import logging

def prune(args, model, logger=None, prune_types=['inter', 'self']):
    if isinstance(args, dict): args = EasyDict(args)
    if hasattr(model, 'module'): model = model.module
    
    for tp in prune_types:
        if tp == 'inter':
            pruning_ratio = args.inter_pruning_ratio or args.pruning_ratio
            pruning_ratio = int(args.hidden_size * args.mlp_ratio * pruning_ratio)
            pruning_method = args.inter_pruning_method
            layer_type = Mlp
        elif tp == 'self':
            pruning_ratio = args.self_pruning_ratio or args.pruning_ratio
            pruning_ratio = int(args.num_heads * pruning_ratio)
            pruning_method = args.self_pruning_method
            layer_type = Attention

        layers = [m for m in model.modules() if isinstance(m, layer_type)]
        slimming_coefs = [m.slimming_coef.detach().cpu().numpy().reshape(-1) for m in layers]

        if args.pruning_strategy == 'random':
            slimming_coefs = [np.random.rand(*coef.shape) for coef in slimming_coefs]
        elif args.pruning_strategy == 'large':
            slimming_coefs = [-coef for coef in slimming_coefs]
        
        if pruning_method == 'global':
            # threshold = np.quantile(np.concatenate(slimming_coefs), pruning_ratio)
            threshold = np.sort(np.concatenate(slimming_coefs))[pruning_ratio]
            threshold = [threshold] * len(slimming_coefs)
        elif pruning_method == 'layerwise':
            # threshold = [np.quantile(coef, pruning_ratio) for coef in slimming_coefs]
            threshold = [np.sort(coef)[min(pruning_ratio, len(coef) - 2)] for coef in slimming_coefs]
        else: assert False

        for m, coef, thre in zip(layers, slimming_coefs, threshold):
            prune_indice = np.where(coef <= thre)[0]
            if logger: 
                logger.warning('Pruning {}: {}, {}'.format(tp, len(prune_indice), prune_indice[:10]))
            else:
                logging.warning('Pruning {}: {}, {}'.format(tp, len(prune_indice), prune_indice[:10]))
            m.prune(prune_indice)

    return model

def calculate_l1_loss(model, tp):
    assert tp in ['inter', 'self']
    layer = Mlp if tp == 'inter' else Attention
    loss = 0.0
    for m in model.modules():
        if isinstance(m, layer) and m.slimming:
            loss += m.slimming_coef.abs().sum()
    return loss

def count_flops(model):
    if hasattr(model, 'module'): model = model.module
    batch_size, n_txt_tokens = 1, 40
    img_dims = (3, 384, 576)
    inputs = {}
    inputs['text_ids'] = torch.ones(batch_size, n_txt_tokens, dtype=torch.int64)
    inputs['text_labels'] = torch.ones(batch_size, n_txt_tokens, dtype=torch.int64)
    inputs['text_masks'] = torch.ones(batch_size, n_txt_tokens, dtype=torch.int64)
    inputs['image'] = [torch.ones(batch_size, *img_dims, dtype=torch.float32)]

    # model = deepcopy(model)
    # model.to('cpu')
    model.current_tasks = []
    flops, params = profile(model, {'batch': inputs}, verbose=False) # one mul-add is counted as 1 flop
    params = sum(p.numel() for n, p in model.named_parameters())
    flops = round(flops / 1e9, 2)
    params = round(params / 1e6, 2)
    return flops, params