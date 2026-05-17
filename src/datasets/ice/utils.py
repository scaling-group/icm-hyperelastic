from omegaconf import ListConfig

GeometryRawCfg = str | list[str]
ForceTypeRawCfg = str | list[str]
ExperimentRawCfg = str | list[str]


class GeometryConfig:
    def __init__(self, cfg: GeometryRawCfg) -> None:
        self.cfg = cfg

        if isinstance(cfg, str):
            self.geometry_ids = [cfg]
        elif isinstance(cfg, list | ListConfig):
            self.geometry_ids = cfg
        else:
            raise TypeError(f"Invalid geometry config: {cfg}")

    def __str__(self) -> str:
        return "".join(self.geometry_ids)


class ForceTypeConfig:
    def __init__(self, cfg: ForceTypeRawCfg) -> None:
        self.cfg = cfg

        if isinstance(cfg, str):
            self.force_type_ids = [cfg]
        elif isinstance(cfg, list | ListConfig):
            self.force_type_ids = cfg
        else:
            raise TypeError(f"Invalid force type config: {cfg}")

    def __str__(self) -> str:
        return "".join(self.force_type_ids)


class ExperimentConfig:
    def __init__(self, cfg: ExperimentRawCfg) -> None:
        self.cfg = cfg

        if isinstance(cfg, str):
            self.experiment_ids = [cfg]
        elif isinstance(cfg, list | ListConfig):
            self.experiment_ids = cfg
        else:
            raise TypeError(f"Invalid experiment config: {cfg}")

    def __str__(self) -> str:
        return "".join(self.experiment_ids)
