import torch
from einops import rearrange, reduce, repeat

# typing
from torch import nn

from src.datasets import ice_data_multi_utils as idu


class EquilUnitEncoder(nn.Module):
    def __init__(self, pool: str, cls_mask: str | None, input_dim: int, projector: nn.Module, transformer: nn.Module):
        super().__init__()

        self.pool = pool
        self.cls_mask = cls_mask
        self.input_dim = input_dim
        self.transformer = transformer
        self.projector = projector

        if self.pool == "cls":
            self.pool_token = nn.Parameter(torch.randn(self.input_dim))

    def forward(self, x: idu.DataEqn) -> torch.Tensor:
        # x: DataEqn
        mask = ~x["mask"]  # (bs, units, max_elements)
        # (bs, units, max_elements+1), zero means unmasked in torch transformer
        if self.pool == "cls" and self.cls_mask == "zero":
            mask = torch.cat([torch.zeros_like(mask[..., -1:]), mask], dim=-1)
        elif self.pool == "cls" and self.cls_mask == "one":
            mask = torch.cat([torch.ones_like(mask[..., -1:]), mask], dim=-1)
        else:
            pass  # no change to mask

        # Flatten A and concatenate with XI
        A_flatten = rearrange(x["A"], "b u m ... -> b u m (...)")  # (bs, units, max_elements, 4)
        prompt = torch.cat([A_flatten, x["XI"]], dim=-1)  # (bs, units, max_elements, 6)

        # Reshape for processing
        prompt = rearrange(prompt, "b u m d -> (b u) m d")
        mask = rearrange(mask, "b u m -> (b u) m")

        # concatenate pool token to the prompt
        if self.pool == "cls":
            pool_token = repeat(self.pool_token, "d -> (b u) 1 d", b=x["A"].size(0), u=x["A"].size(1))
            prompt = torch.cat([pool_token, prompt], dim=1)  # (bs*units, max_elements+1, 6)
        else:
            pass  # no change to prompt

        # Project and encode
        embedding = self.projector(prompt)  # (bs*units, max_elements+1, model_dim)
        embedding = self.transformer(embedding, src_key_padding_mask=mask)

        # Extract pool token output and reshape
        if self.pool == "cls":
            embedding = embedding[:, 0, :]  # (bs*units, model_dim)
            embedding = rearrange(embedding, "(b u) d -> b u d", b=x["A"].size(0))
        elif self.pool == "average":
            embedding = reduce(embedding, "x m d -> x d", "mean")
            embedding = repeat(embedding, "(b u) d -> b u d", b=x["A"].size(0))

        return embedding


class ICE_EnDecoder(nn.Module):
    def __init__(
        self,
        preprocessor: nn.Module,
        endecoder: nn.Module,
        query_projector: nn.Module,
        post_projector: nn.Module,
        params: dict,
    ):
        """
        Encoder-Decoder model for ICON
        """
        super().__init__()

        self.preprocessor = preprocessor
        self.endecoder = endecoder
        self.query_projector = query_projector
        self.post_projector = post_projector

    def forward(self, prompt: idu.DataEqn, query: torch.Tensor, **kwargs):
        """
        since we have reshaped query to (bs, len, d) before model call, we don't need to rearrange it here
        return shape: (bs, len, d)
        """
        memory = self.preprocessor(prompt)
        query = self.query_projector(query)
        # query = rearrange(query, "b ... d -> b (...) d")
        _, query_output = self.endecoder(memory=memory, tgt=query)
        output = self.post_projector(query_output)
        # output = output.view(output.shape[0], *query.shape[1:-1], output.shape[-1])
        return output  # (bs, ..., output_dim)


if __name__ == "__main__":
    # run by `uv run python -m src.models.ice.pre_endecoder`
    import hydra
    import rootutils
    from lightning.fabric.utilities.throughput import measure_flops
    from omegaconf import DictConfig

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    @hydra.main(version_base="1.3", config_path="../../../configs/", config_name="train_ice.yaml")
    def main(cfg: DictConfig):
        bs = 32
        num_eqn = 470  # this will approximate the FLOP in training

        with torch.device("meta"):
            prompt = {
                "A": torch.randn(bs, num_eqn, 10, 2, 2),
                "XI": torch.randn(bs, num_eqn, 10, 2),
                "mask": torch.ones(bs, num_eqn, 10).bool(),
            }
            query = torch.randn(bs, num_eqn * 10, 2)
            model = hydra.utils.instantiate(cfg.model)

        num_params = sum(param.numel() for param in model.parameters())
        print(f"{num_params:,} parameters in total")

        # https://lightning.ai/docs/fabric/2.5.2/api/utilities.html#lightning.fabric.utilities.throughput.measure_flops
        def model_fwd():
            return model(prompt, query)

        def model_loss(y):
            return y.sum()

        flops = measure_flops(model, model_fwd)
        print(f"Forward pass FLOPs: {flops:,}")
        flops = measure_flops(model, model_fwd, model_loss)
        print(f"Loss FLOPs: {flops:,}")

    main()
