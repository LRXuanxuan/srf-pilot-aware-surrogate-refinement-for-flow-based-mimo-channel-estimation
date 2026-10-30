# Naming and migration map

This document maps every legacy artifact (the `MIMO_gm2/` research snapshot) to
its home in the release package and records what was intentionally dropped. It
also lists the legacy inconsistencies that were resolved while extracting the
package.

## 1. Files

| Legacy (`MIMO_gm2/`) | New home | Notes |
| --- | --- | --- |
| `models.py` | `src/srf/models.py` | `UNet` renamed `GMUNet`; topology unchanged. Unused `CNN` and `get_positional_embedding` dropped. `SinusoidalPosEmb`, `Downsample`, `Upsample`, `Block`, `ResnetBlock` kept. |
| `gm_utils.py` | `src/srf/gm.py` | One canonical implementation per helper; duplicate `sample_gm` collapsed (the second, shared-logstd-first definition is kept). `normalize_gm_logweights` dropped (unused). |
| `loader.py` | `src/srf/data.py` | `MatChannelDataset` behaviour preserved; `ChannelConfig` and `save_mat_dataset` added for scripts/tests; `hdf5storage` imported lazily; console prints removed; samples are returned as `torch.Tensor` instead of NumPy arrays (same shapes/dtypes after default collation). |
| `utils.py` | `src/srf/utils.py` | `get_dynamic_nfe` unchanged; dtype/device helpers added for fp16-safe inference. |
| `test_rcflow_fig_gm1tr05_multi.py` | `src/srf/sampler.py`, `src/srf/physics.py`, `src/srf/gm.py`, `src/srf/training.py` | Source of truth for the SRF/SRF-P sampler. `rcflow_inference` -> `sample_channels`; `_apply_fig_guidance` -> `_apply_pbg_guidance`; diagnostics keys `fig_*` -> `pbg_*`; return value `(h, nfe1, nfe2, diag)` -> `(h, diag)` with `nfe1`/`nfe2` inside `diagnostics`. |
| `test_rcflow.py` | `src/srf/baselines/rcflow.py` | RC-Flow baseline; `rcflow_inference` -> `rcflow_sample_channels`. |
| `test_flower_full_gm1tr05.py` | `src/srf/baselines/flower.py` | `flower_single_trajectory` -> `_single_trajectory`; `flower_full_inference` -> `flower_sample_channels`; `FLOWER_MODES` exported. |
| `train_flow.py`, `train_flow_gm1tr05_multi.py`, `train_flow_flower_gm1tr05.py` | `src/srf/training.py` | `GMFlowMatchingLoss` -> `GMFlowObjective`; `GM1TransitionNLLLoss` folded into the same objective. The `accelerate` loop, EMA and checkpoint saving move to the Lead-owned `scripts/train.py` (no `accelerate` dependency in `src/`). |
| `test_rcflow_fig.py`, `test_rcflow_baseline_fig_compatible.py` | dropped | Near-duplicates of the `gm1tr05` sampler core; they differ only in the PBG defaults (`FIG_C` 0.10/0.15, `FIG_PLUS_MIX` 0.08/0.20) and in the default checkpoint path (tr = 0.5 vs tr = 1.0). The release keeps the 0.10 / 0.08 defaults. |
| `test_rcflow_fig_grid.py` | dropped | Internal hyper-parameter grid search. |
| `test_rcflow_fig_prefix_sweep.py` | dropped | Internal test-prefix sweep. |
| `test_rcflow_batch_csv.py`, `test_flower_full_batch_csv.py` | dropped | Internal CSV evaluation harnesses; evaluation is provided by the Lead-owned scripts. |
| `rcflow_core.py` | dropped | Broken half-finished refactor (imports physics helpers that do not exist in `gm_utils`); used only as a style hint. Its `original_rc_prox_update` / `rc_prior_variance_from_time` semantics live in `srf.physics.proximal_update`. |
| legacy `README.md`, `run_gm1tr05_experiments.sh` | dropped | Replaced by the Lead-owned `README.md` and `scripts/`. |
| `__pycache__/`, `.ipynb_checkpoints/` | dropped | Build/Jupyter artifacts. |

## 2. Terms and symbols

| Legacy term | New name | Meaning |
| --- | --- | --- |
| `SAMPLER_MODE = "baseline"` | `SamplerConfig(mode="srf_p")` | SRF-P: DPR centre + closed-form proximal update, no PBG. |
| `SAMPLER_MODE = "fig"` | `SamplerConfig(mode="pbg_only")` | PBG guidance without null/range recombination (ablation). |
| `SAMPLER_MODE = "fig_plus"` | `SamplerConfig(mode="srf")` | Full SRF: PBG + null/range mixing with `rho` (paper Algorithm 1). |
| RC-Flow baseline (`test_rcflow.py`, tr = 1.0) | `SamplerConfig(mode="rcflow")` | Point-velocity sampler. |
| FLOWER baseline (`test_flower_full_gm1tr05.py`) | `SamplerConfig(mode="flower")` / `srf.baselines.flower` | Full-posterior comparison sampler; the mode delegates to `flower_sample_channels` (`gm_output_mode` selects `mean`/`sample`). |
| `FIG` | `PBG` | Posterior bridge guidance, paper eqs. (29)-(36). |
| `gm1tr05` / `gm1_tr05` | `DPR` | GM `K = 1`, finite-transition ratio `r = 0.5`. |
| `FIG_K` | `pbg_iters` | PBG iterations per reverse step (default 1). |
| `FIG_C` / `fig_c` | `pbg_step` | PBG step scale (default 0.10). |
| `FIG_W` | `pbg_w` | Weight of the noise term in the PBG pilot interpolant (default 1.0). |
| `FIG_PLUS_MIX` / `fig_plus_mix` | `pbg_rho` | Null/range recombination weight of full SRF (default 0.08). |
| `FIG_NOISE_COEF` | `pbg_noise_coef` | Noise gate coefficient (default 1.0). |
| `FIG_EPS` | `pbg_eps` | Numerical epsilon (default 1e-6). |
| `FIG_SCALE_MAX` | `pbg_step_max` | PBG step upper bound (default 0.5). |
| `LAMBDA` | `lambda_exp` | Reverse-time warp exponent of `t_curr` (= 2). |
| `BETA` | `beta_exp` | Reverse-time warp exponent of `t_next` (= 2). |
| `GM_OUTPUT_MODE` | `gm_output_mode` | `"mean"` or `"sample"` DPR centre. |
| `TRANSITION_RATIO` | `GMFlowObjective(transition_ratio=...)` | Finite-transition ratio `r` of DPR training (0.5 in the paper). |
| `TEST_DATASET_TYPE` / `TRAIN_DATASET_TYPE` | Lead-owned config/CLI | CDL type selection. |
| `gm_utils.gm_to_mean` | `gm.gm_to_mean` | Mixture mean. |
| `gm_utils.gm_second_moment` | `gm.gm_second_moment` | Mixture second moment. |
| `gm_utils.gm_effective_variance` | `gm.gm_effective_variance` | Effective isotropic variance. |
| `gm_utils.sample_gm` | `gm.sample_gm` | Single sample from the mixture. |
| `gm_utils.u_to_x0_gm` | `gm.u_to_x0_gm` | Velocity mixture -> clean-channel mixture (paper eq. (25) centre). |
| `gm_utils.reverse_transition_additive` | `gm.reverse_transition_additive` | Finite-transition target for DPR training. |
| `gm_utils.gm_logprob` / `gm_nll_loss` | `gm.gm_logprob` / `gm.gm_nll_loss` | Mixture log-density / NLL. |
| `gm_utils._expand_shared_logstd` | `gm.expand_shared_logstd` | Broadcast a shared log-std. |
| local `gm_global_stats` in test scripts | `gm.gm_global_stats` | Mixture entropy / max weight / log-std stats. |
| local `gm_tau2_from_x0_gm` in test scripts | `gm.gm_tau2_from_x0_gm` | Learned clean-channel variance proxy. |
| `build_physics_cache` | `physics.build_physics_cache` | Keys unchanged: `sigma2`, `p_h`, `p_p_h`, `p_pinv`, `proj`, `fig_lipschitz`, `pilot_ratio`, `y_p_h`, `i_nt`, `y_cplx`, `p_cplx`. |
| `apply_measurement` | `physics.apply_measurement` | `H -> H P`. |
| `apply_range_projection` | `physics.apply_range_projection` | `H P P^dagger`. |
| `apply_null_projection` | `physics.apply_null_projection` | `H - H P P^dagger`. |
| `fig_measurement_interpolant` | `physics.pbg_target` | PBG pilot-space target. |
| `fig_data_consistency_gradient` | `physics.pbg_gradient` | PBG residual gradient (operator-normalised). |
| inline proximal update in `rcflow_inference` | `physics.proximal_update` | Closed-form pilot-consistency step, paper eq. (27). |
| `load_model_checkpoint` / `_strip_module_prefix` / `_extract_state_dict` | `training.load_checkpoint` / `training.strip_state_dict_prefix` / `training.extract_state_dict` | Checkpoint helpers. |
| `GMFlowMatchingLoss` | `training.GMFlowObjective` | DPR objective. |
| classic flow-matching MSE (not present in the legacy snapshot; reconstructed for the RC-Flow baseline) | `training.FlowMatchingObjective` | RC-Flow baseline objective. |
| `flower_full_inference` | `baselines.flower.flower_sample_channels` | FLOWER comparison sampler. |

Diagnostics keys previously prefixed `fig_` are now `pbg_` (`pbg_residual_mean`,
`pbg_grad_norm_mean`, `pbg_step_mean`, `pbg_lipschitz_mean`); all other
diagnostics keys are unchanged.

## 3. Intentionally dropped

* Hard-coded server paths (`/home/featurize/...`), checkpoint names and output
  directories; all paths are now caller-provided.
* Chinese comments and docstrings; the release is English-only.
* Dead configuration: `GAMMA` (never used) and `ADAPTIVE_P` (kept for
  compatibility but not part of the GM NLL objective).
* Legacy mode aliases `base`, `gm_mean_rc`, `mean_rc`, `fig+` (the canonical
  aliases `baseline`, `fig`, `fig_plus` are still accepted).
* Duplicated implementations that existed both in `gm_utils.py` and inside the
  test scripts; `srf.gm` / `srf.physics` are now the only copies.
* `__pycache__/`, `.ipynb_checkpoints/`, old checkpoints, grid-search and
  CSV-harness scripts, and the legacy shell runner.

## 4. Legacy inconsistencies resolved

1. **Duplicate `sample_gm`.** `gm_utils.py` defined `sample_gm` twice; the
   second definition (shared log-std detected before per-component log-std)
   is the canonical one and is kept in `srf.gm.sample_gm`.
2. **Duplicate physics/GM helpers in test scripts.** The same functions were
   copied into several test files; they are now single implementations in
   `srf.gm` and `srf.physics`.
3. **Broken `rcflow_core.py`.** It imported physics helpers from `gm_utils`
   that never existed; it is dropped and its intent is covered by
   `srf.physics.proximal_update`.
4. **`train_flow.py` drift.** The snapshot's `train_flow.py` no longer contains
   the classic RC-Flow MSE objective (it holds the GM NLL objective with
   `K = 2`, `tr = 0.5`), although the legacy README still describes it as the
   RC-Flow trainer. `training.FlowMatchingObjective` reconstructs the classic
   additive-path point-velocity MSE; `training.GMFlowObjective` is the actual
   GM NLL objective used for DPR.
5. **Time sampler default versus entry point.** `GMFlowMatchingLoss.__init__`
   defaults to `time_mu = -0.4`, but the legacy training entry points passed
   `TIME_MU = 0.0` (and the paper does not specify the departure time). The
   release uses `time_mu = 0.0` for reproducibility with the legacy runs; pass
   `--time-mu`/`time_mu` in the config to override.
6. **PBG default drift.** `test_rcflow_fig.py` used `FIG_C = 0.15` and
   `FIG_PLUS_MIX = 0.20`, while the reference `gm1tr05` script used 0.10 and
   0.08. The release standardises on the paper defaults 0.10 / 0.08.
7. **Checkerboard complex noise initialisation.** The legacy SRF sampler
   initialises the latent with
   `(randn_like(x) + 1j * randn_like(x)) / sqrt(2)` where `x` is complex. This
   is *not* a circularly symmetric `CN(0, 1)` draw (the real and imaginary
   parts get different variances). The expression is reproduced verbatim in
   `srf.sampler._legacy_complex_noise` so that inference matches the reference
   numbers; the FLOWER baseline keeps its own (proper) `CN(0, 1)` generator,
   as in the legacy code.
8. **`test_rcflow.py` checkpoint loading.** It called `torch.load` directly and
   assumed a raw state dict, while other scripts supported wrapped checkpoints.
   `training.load_checkpoint` handles both layouts (and checkpoints with
   metadata) in one place.
9. **Unused `fig_k <= 0` and `GAMMA` branches.** Kept only where they change
   behaviour (`pbg_iters = 0` disables PBG); `GAMMA` is dropped.
