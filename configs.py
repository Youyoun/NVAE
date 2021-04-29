from dataclasses import dataclass


@dataclass
class EncoderConfig:
    n_channels: int
    n_preprocess_blocks: int  # block is defined as series of Normal followed by Down
    n_preprocess_cells: int  # number of cells per block
    n_cell_per_cond: int  # number of cell for each conditional in encoder


@dataclass
class DecoderConfig:
    n_channels: int
    n_postprocess_blocks: int
    n_postprocess_cells: int
    n_cell_per_cond: int  # number of cell for each conditional in decoder
    n_mix_output: int = 10


@dataclass
class NVAEConfig:
    n_latent_scales: int  # number of spatial scales that latent layers will reside
    n_groups_per_scale: int  # number of groups of latent vars. per scale
    n_latent_per_group: int  # number of latent vars. per group
    min_groups_per_scale: int
    ada_groups: int
    dataset: str
    use_se: bool
    res_dist: int
    encider: EncoderConfig
    decoder: DecoderConfig
