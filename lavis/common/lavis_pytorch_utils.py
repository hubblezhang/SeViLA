"""lavis_pytorch_utils.py
统一提供 lavis 中使用的 transformers 内部工具函数。
新版 transformers（4.30+）把 apply_chunking_to_forward / find_pruneable_heads_and_indices /
prune_linear_layer / ALL_LAYERNORM_LAYERS 从 modeling_utils / pytorch_utils 中移走了。
这里提供一个稳定入口：先尝试从新老 transformers 路径 import，都找不到就自己 inline 定义。
"""
import torch
from torch import nn
from torch.nn import LayerNorm


# ============================================================
# apply_chunking_to_forward
# ============================================================
def _chunker_fn(forward_fn, chunk_size, chunk_dim, *input_tensors):
    assert len(input_tensors) > 0
    assert chunk_size is not None and chunk_dim is not None
    if torch.jit.is_tracing():
        return forward_fn(*input_tensors)
    assert all(t.shape[chunk_dim] == input_tensors[0].shape[chunk_dim] for t in input_tensors)
    num_chunks = max(1, input_tensors[0].shape[chunk_dim] // chunk_size)
    if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
        num_chunks += 1
    if num_chunks == 1:
        return forward_fn(*input_tensors)
    chunked_input_tensors = [list(t.chunk(num_chunks, chunk_dim)) for t in input_tensors]
    outputs = [forward_fn(*t_list) for t_list in zip(*chunked_input_tensors)]
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=chunk_dim)
    elif isinstance(outputs[0], (list, tuple)):
        return type(outputs[0])(
            torch.cat([out[i] for out in outputs], dim=chunk_dim)
            for i in range(len(outputs[0]))
        )
    else:
        raise TypeError(
            f"forward_fn returns {type(outputs[0])} but should return Tensor or list/tuple"
        )


def _import_apply_chunking():
    try:
        from transformers.modeling_utils import apply_chunking_to_forward as _f
        return _f
    except Exception:
        pass
    try:
        from transformers.pytorch_utils import apply_chunking_to_forward as _f
        return _f
    except Exception:
        pass
    return _chunker_fn


# ============================================================
# find_pruneable_heads_and_indices
# ============================================================
def _find_pruneable(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _import_find_pruneable():
    try:
        from transformers.modeling_utils import find_pruneable_heads_and_indices as _f
        return _f
    except Exception:
        pass
    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices as _f
        return _f
    except Exception:
        pass
    return _find_pruneable


# ============================================================
# prune_linear_layer
# ============================================================
def _prune_linear(layer, index, dim=0):
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if getattr(layer, "bias", None) is not None:
        b = layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device, dtype=layer.weight.dtype
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if getattr(layer, "bias", None) is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def _import_prune_linear():
    try:
        from transformers.modeling_utils import prune_linear_layer as _f
        return _f
    except Exception:
        pass
    try:
        from transformers.pytorch_utils import prune_linear_layer as _f
        return _f
    except Exception:
        pass
    return _prune_linear


# ============================================================
# ALL_LAYERNORM_LAYERS (一个值，不是函数)
# ============================================================
def _import_all_layernorm():
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS as _a
        return list(_a)
    except Exception:
        pass
    try:
        from transformers.modeling_utils import ALL_LAYERNORM_LAYERS as _a
        return list(_a)
    except Exception:
        pass
    return [LayerNorm]


# ============================================================
# 导出
# ============================================================
_apply_chunking_to_forward = _import_apply_chunking()
_find_pruneable_heads_and_indices = _import_find_pruneable()
_prune_linear_layer = _import_prune_linear()
_ALL_LAYERNORM_LAYERS = _import_all_layernorm()


def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    return _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors)


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    return _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads)


def prune_linear_layer(layer, index, dim=0):
    return _prune_linear_layer(layer, index, dim)


# 注意：ALL_LAYERNORM_LAYERS 是一个 list，不是函数 —— modeling_t5 里把它当值用
ALL_LAYERNORM_LAYERS = _ALL_LAYERNORM_LAYERS
