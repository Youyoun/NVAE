from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset: str
    data: str
    root: str
    save: str


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
class NormalizingFlowParameters:
    num_nf: int
    n_bits: int


@dataclass
class LatentSpaceConfig:
    n_latent_scales: int  # number of spatial scales that latent layers will reside
    n_groups_per_scale: int  # number of groups of latent vars. per scale
    n_latent_per_group: int  # number of latent vars. per group
    min_groups_per_scale: int
    ada_groups: bool  # different number of groups per scale


@dataclass
class NVAEConfig:
    use_se: bool
    res_dist: bool
    norm_flow: NormalizingFlowParameters
    latent: LatentSpaceConfig
    encoder: EncoderConfig
    decoder: DecoderConfig


@dataclass
class OptimizationParameters:
    batch_size: int
    epochs: int
    warmup_epochs: int
    learning_rate: float
    learning_rate_min: float
    weight_decay: float
    weight_decay_norm: float
    weight_decay_norm_init: float
    weight_decay_norm_anneal: float


@dataclass
class KLAnnealingParameters:
    kl_anneal_portion: float
    kl_const_portion: float
    kl_const_coeff: float


@dataclass
class DistributedConfig:
    distributed: bool
    num_proc_node: int
    node_rank: int
    local_rank: int
    global_rank: int
    num_process_per_node: int
    master_address: int
