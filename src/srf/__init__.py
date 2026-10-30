"""SRF: pilot-aware surrogate refinement for flow-based MIMO channel estimation.

Reference implementation of the SRF estimator of

    "SRF: Pilot-Aware Surrogate Refinement for Flow-Based MIMO Channel Estimation"

The package is organised as follows:

* :mod:`srf.models`   -- convolutional flow prior with a Gaussian-mixture head;
* :mod:`srf.gm`       -- Gaussian-mixture mathematics;
* :mod:`srf.physics`  -- pilot-space measurement/projection operators and the
  closed-form proximal update (paper eq. (27));
* :mod:`srf.sampler`  -- SRF / SRF-P / PBG-only / RC-Flow samplers;
* :mod:`srf.baselines`-- RC-Flow and FLOWER comparison methods;
* :mod:`srf.data`     -- ``.mat`` channel dataset and pilot generation;
* :mod:`srf.training` -- DPR and RC-Flow training objectives;
* :mod:`srf.utils`    -- adaptive step-count schedule.
"""

from .baselines import FLOWER_MODES, flower_sample_channels, rcflow_sample_channels
from .data import ChannelConfig, MatChannelDataset, save_mat_dataset
from .gm import (
    expand_shared_logstd,
    gm_effective_variance,
    gm_global_stats,
    gm_logprob,
    gm_nll_loss,
    gm_second_moment,
    gm_tau2_from_x0_gm,
    gm_to_mean,
    reverse_transition_additive,
    sample_gm,
    u_to_x0_gm,
)
from .models import GMUNet
from .physics import (
    apply_measurement,
    apply_null_projection,
    apply_range_projection,
    build_physics_cache,
    pbg_gradient,
    pbg_target,
    proximal_update,
)
from .sampler import SamplerConfig, sample_channels
from .training import (
    FlowMatchingObjective,
    GMFlowObjective,
    load_checkpoint,
    strip_state_dict_prefix,
)
from .utils import get_dynamic_nfe

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Models
    "GMUNet",
    # Samplers
    "SamplerConfig",
    "sample_channels",
    "FLOWER_MODES",
    "flower_sample_channels",
    "rcflow_sample_channels",
    # Data
    "ChannelConfig",
    "MatChannelDataset",
    "save_mat_dataset",
    # Training
    "FlowMatchingObjective",
    "GMFlowObjective",
    "load_checkpoint",
    "strip_state_dict_prefix",
    # Gaussian mixtures
    "expand_shared_logstd",
    "gm_effective_variance",
    "gm_global_stats",
    "gm_logprob",
    "gm_nll_loss",
    "gm_second_moment",
    "gm_tau2_from_x0_gm",
    "gm_to_mean",
    "reverse_transition_additive",
    "sample_gm",
    "u_to_x0_gm",
    # Physics / PBG
    "apply_measurement",
    "apply_null_projection",
    "apply_range_projection",
    "build_physics_cache",
    "pbg_gradient",
    "pbg_target",
    "proximal_update",
    # Utilities
    "get_dynamic_nfe",
]
