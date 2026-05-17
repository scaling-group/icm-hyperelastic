import einops
import optree
import torch
from omegaconf import DictConfig
from optree import PyTree

from src.datasets import ice_data_multi_utils as idu

## pre-processing


def scale_prompt_A(raw_prompt_A: torch.Tensor, mode: str | None) -> torch.Tensor:
    """
    Scale the prompt A
    raw_prompt_A: (bs, n_eq, N, 2, 2)
    """
    if mode is None:
        return raw_prompt_A
    elif mode == "norm":
        scale = einops.reduce(raw_prompt_A**2, "bs n_eq N d1 d2 -> bs n_eq 1 d1 1", "sum").sqrt()
        # (bs, nfree, 1, 2, 1)
        scaled_prompt_A = raw_prompt_A / scale
    else:
        raise ValueError(f"Got unknown mode: {mode}")
    return scaled_prompt_A


def scale_XI(
    raw_prompt_XI: torch.Tensor,
    raw_prompt_mask: torch.Tensor,
    raw_queries: PyTree,
    mode: list[str],
) -> tuple[torch.Tensor, PyTree]:
    """
    Scale the XI, including the prompt and queries
    raw_prompt_XI: (bs, n_eq, N, 2)
    raw_prompt_mask: (bs, n_eq, N)
    raw_queries: PyTree, each leaf of shape (bs, seq, 2)
    mode is a list of strings. We will apply the operation sequentially as a chain.
    """
    center = torch.tensor([2.0, 1.0]).to(raw_prompt_XI)

    prompt_XI = raw_prompt_XI
    queries = raw_queries

    for op in mode:
        if op == "decenter":  # XI = XI - center

            def operate(x):
                return x - center
        elif op == "std":
            mask = raw_prompt_mask[..., None]  # (bs, n_eq, N, 1)
            x = prompt_XI * mask  # (bs, n_eq, N, 2), cannot skip, since XI may not be (0,0) if mask=0
            n_valid = mask.sum(dim=(-2, -3), keepdim=True)  # (bs, 1, 1, 1)
            mean = x.sum(dim=(-2, -3), keepdim=True) / n_valid  # (bs, 1, 1, 2), mean of non-padding elements
            # (bs, 1, 1, 2), variance of non-padding elements
            var = ((x - mean) ** 2).sum(dim=(-2, -3), keepdim=True) / n_valid
            scale = torch.sqrt(var)  # (bs, 1, 1, 2), std of non-padding elements

            # captures the current value of scale at the time the function is defined
            def operate(x, scale=scale):
                return x / scale
        elif op == "zstd":  # zero centered std
            mask = einops.rearrange(raw_prompt_mask, "bs n_eq N -> bs n_eq N 1")  # (bs, n_eq, N, 1)
            x = prompt_XI * mask  # (bs, n_eq, N, 2), cannot skip, since default XI may not be (0,0) if mask=0
            n_valid = einops.reduce(mask, "bs n_eq N 1 -> bs 1 1 1", "sum")
            sum_square = einops.reduce(x**2, "bs n_eq N d -> bs 1 1 d", "sum")
            scale = torch.sqrt(sum_square / n_valid)  # (bs, 1, 1, 2), std of non-padding elements

            # captures the current value of scale at the time the function is defined
            def operate(x, scale=scale):
                return x / scale
        elif op == "max":
            mask = raw_prompt_mask[..., None]  # (bs, n_eq, N, 1)
            x = prompt_XI * mask  # (bs, n_eq, N, 2), cannot skip, since default XI may not be (0,0) if mask=0
            scale = einops.reduce(x.abs(), "bs n_eq N d -> bs 1 1 d", "max")  # (bs, 1, 1, 2)

            # captures the current value of scale at the time the function is defined
            def operate(x, scale=scale):
                return x / scale
        elif op == "log1p":  # if x < 0, return x; if x >= 0, return log(1+x)

            def operate(x):
                return x * (x < 0) + torch.log(torch.clamp_min(1.0 + x, 0.5)) * (x >= 0)
        elif op == "encenter":  # XI = XI + center

            def operate(x):
                return x + center
        else:
            raise ValueError(f"Got unknown operation: {op}")
        prompt_XI = operate(prompt_XI)
        queries = optree.tree_map(operate, queries)

    return prompt_XI, queries


def scale_prompt_queries(raw_prompt: idu.DataEqn, raw_queries: PyTree, cfg: DictConfig) -> tuple[idu.DataEqn, PyTree]:
    """
    Scale the prompt and queries
    We will calculate the scale from raw_prompt,
    and apply the same scale to each leaf in raw_queries, as well as raw_prompt["XI"]
    """
    # scaling for A
    scaled_prompt_A = scale_prompt_A(raw_prompt["A"], cfg.loss.scale_A)
    scaled_prompt_XI, scaled_queries = scale_XI(raw_prompt["XI"], raw_prompt["mask"], raw_queries, cfg.loss.scale_XI)
    scaled_prompt = idu.DataEqn(A=scaled_prompt_A, XI=scaled_prompt_XI, mask=raw_prompt["mask"])
    return scaled_prompt, scaled_queries


class QueryProcessor:
    def __init__(self, queries: PyTree):
        self.queries = queries
        self.structure = optree.tree_structure(queries)
        self.split_sizes = None

    def flatten(self):
        """
        convert the query PyTree to a tensor of shape (bs, len, d)
        step 1: flatten the query PyTree to a list
        step 2: reshape each element in the list to (bs, (...), d)
        step 3: concat the reshaped elements along the sequence dimension
        """
        # Flatten the PyTree to a list of tensors
        tensorlist, _ = optree.tree_flatten(self.queries)
        # Process each tensor to ensure it has the right shape
        reshaped_tensors = [einops.rearrange(tensor, "b ... d -> b (...) d") for tensor in tensorlist]
        self.split_sizes = [tensor.shape[1] for tensor in reshaped_tensors]
        # Concatenate along sequence dimension
        return torch.cat(reshaped_tensors, dim=1)

    def unflatten(self, flattened_tensor: torch.Tensor):
        """
        convert the flattened tensor back to the original PyTree
        step 1: split the flattened tensor along the sequence dimension
        step 2: reshape each element in the list to (bs, ..., d)
        step 3: unflatten the list to the original PyTree
        """
        # split the flattened tensor
        split_tensors = torch.split(flattened_tensor, self.split_sizes, dim=1)
        # get the PyTree structure
        tensor_pytree = optree.tree_unflatten(self.structure, split_tensors)
        # the target shape of tensor in the PyTree
        shape = optree.tree_map(lambda x: [*x.shape[:-1], flattened_tensor.shape[-1]], self.queries)
        # reshape the tensor in the PyTree to the target shape
        return optree.tree_map(lambda x, s: x.reshape(s), tensor_pytree, shape)
