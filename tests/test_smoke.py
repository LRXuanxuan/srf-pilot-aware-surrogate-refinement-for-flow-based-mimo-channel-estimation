"""Smoke tests for the SRF release package.

The tests use a deliberately tiny configuration (``N_r = 2``, ``N_t = 4``,
``N_p = 2``, batch of 2 and two reverse steps) so that the whole sampler runs on
CPU in a few seconds.  ``torch`` is required at runtime; if it is missing the
whole module is skipped instead of failing.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

torch = pytest.importorskip("torch")

import srf  # noqa: E402  (import after the torch availability check)

BATCH = 2
NR = 2
NT = 4
NPILOTS = 2
NOISE_SCALE = 0.3


def _tiny_model() -> srf.GMUNet:
    torch.manual_seed(0)
    model = srf.GMUNet(
        in_channels=2,
        out_channels=2,
        model_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        gm_num_components=2,
    )
    model.eval()
    return model


def _make_inputs():
    generator = torch.Generator().manual_seed(0)

    def _complex_randn(*shape: int) -> torch.Tensor:
        real = torch.randn(*shape, generator=generator)
        imag = torch.randn(*shape, generator=generator)
        return (real + 1j * imag) / math.sqrt(2.0)

    h_true = _complex_randn(BATCH, NR, NT)
    p_cplx = _complex_randn(BATCH, NT, NPILOTS)
    y_cplx = torch.bmm(h_true, p_cplx)
    noise_scale = torch.full((BATCH,), NOISE_SCALE)
    return h_true, p_cplx, y_cplx, noise_scale


def _gm_tensors():
    generator = torch.Generator().manual_seed(1)
    means = torch.randn(BATCH, 2, 2, NR, NT, generator=generator)
    logstds = torch.randn(BATCH, 1, 1, 1, 1, generator=generator).clamp(-3.0, 0.0)
    logweights = torch.randn(BATCH, 2, 1, NR, NT, generator=generator).log_softmax(dim=1)
    return {"means": means, "logstds": logstds, "logweights": logweights}


def test_version_and_defaults():
    assert srf.__version__ == "0.1.0"
    signature = inspect.signature(srf.GMUNet.__init__)
    assert signature.parameters["gm_num_components"].default == 1
    assert signature.parameters["model_channels"].default == 32
    assert signature.parameters["channel_mult"].default == (1, 2, 4)


def test_gmunet_forward_shapes():
    model = _tiny_model()
    x = torch.randn(BATCH, 2, NR, NT)
    t = torch.rand(BATCH)
    out = model(x, t)
    assert out["means"].shape == (BATCH, 2, 2, NR, NT)
    assert out["logstds"].shape == (BATCH, 1, 1, 1, 1)
    assert out["logweights"].shape == (BATCH, 2, 1, NR, NT)
    assert torch.allclose(
        out["logweights"].logsumexp(dim=1), torch.zeros(BATCH, 1, NR, NT), atol=1e-6
    )
    assert torch.isfinite(out["means"]).all()


def test_gm_helpers():
    gm = _gm_tensors()
    target = torch.randn(BATCH, 2, NR, NT)

    mean = srf.gm_to_mean(gm)
    second = srf.gm_second_moment(gm)
    variance = srf.gm_effective_variance(gm)
    assert mean.shape == (BATCH, 2, NR, NT)
    assert second.shape == mean.shape
    assert (variance > 0).all()

    samples_a = srf.sample_gm(gm, generator=torch.Generator().manual_seed(3))
    samples_b = srf.sample_gm(gm, generator=torch.Generator().manual_seed(3))
    assert samples_a.shape == mean.shape
    assert torch.equal(samples_a, samples_b)

    logprob = srf.gm_logprob(gm, target)
    # Per-pixel log-density keeps the singleton mixture-component axis (legacy layout).
    assert logprob.shape == (BATCH, 1, NR, NT)
    nll = srf.gm_nll_loss(gm, target)
    assert torch.isfinite(nll)

    sigma = torch.full((BATCH,), 0.5)
    gm_x0 = srf.u_to_x0_gm(gm, target, sigma)
    assert gm_x0["means"].shape == (BATCH, 2, 2, NR, NT)
    assert torch.allclose(gm_x0["means"], target.unsqueeze(1) - 0.5 * gm["means"])

    gm_transition = srf.reverse_transition_additive(
        gm, target, torch.full((BATCH,), 0.25), torch.full((BATCH,), 0.5)
    )
    assert torch.allclose(gm_transition["means"], target.unsqueeze(1) - 0.25 * gm["means"])

    expanded = srf.expand_shared_logstd(gm["logstds"], (BATCH, 2, NR, NT))
    assert expanded.shape == (BATCH, 2, NR, NT)

    stats = srf.gm_global_stats(gm)
    assert set(stats) == {"entropy_mean", "max_weight_mean", "logstd_mean"}
    assert srf.gm_tau2_from_x0_gm(gm_x0).shape == (BATCH,)


def test_physics_and_proximal_update():
    _, p_cplx, y_cplx, noise_scale = _make_inputs()
    h_true = torch.randn(BATCH, NR, NT, dtype=torch.complex64)
    cache = srf.build_physics_cache(y_cplx, p_cplx, noise_scale)

    for key in (
        "sigma2",
        "p_h",
        "p_p_h",
        "p_pinv",
        "proj",
        "fig_lipschitz",
        "pilot_ratio",
        "y_p_h",
        "i_nt",
        "y_cplx",
        "p_cplx",
    ):
        assert key in cache
    assert cache["proj"].shape == (BATCH, NT, NT)
    assert cache["i_nt"].shape == (BATCH, NT, NT)
    assert torch.allclose(cache["pilot_ratio"], torch.full_like(cache["pilot_ratio"], NPILOTS / NT))

    range_part = srf.apply_range_projection(h_true, cache)
    null_part = srf.apply_null_projection(h_true, cache)
    assert torch.allclose(range_part + null_part, h_true, atol=1e-5)
    assert srf.apply_measurement(h_true, cache).shape == (BATCH, NR, NPILOTS)

    epsilon = torch.randn(BATCH, NR, NT, dtype=torch.complex64)
    y_interp = srf.pbg_target(
        y_cplx, epsilon, cache, torch.full((BATCH,), 0.5), 1.0
    )
    assert y_interp.shape == y_cplx.shape
    grad = srf.pbg_gradient(h_true, y_interp, cache)
    assert grad.shape == h_true.shape
    assert torch.isfinite(grad).all()

    # paper eq. (27), recomputed independently
    t = 0.5
    vt_sq = t**2 / (t**2 + (1 - t) ** 2 + 1e-6)
    noise_b = noise_scale.view(BATCH, 1, 1)
    lhs = cache["p_p_h"] / noise_b**2 + cache["i_nt"] / (vt_sq + 1e-6)
    rhs = cache["y_p_h"] / noise_b**2 + h_true / (vt_sq + 1e-6)
    expected = torch.bmm(rhs, torch.linalg.solve(lhs, cache["i_nt"]))
    h_proj = srf.proximal_update(y_cplx, h_true, p_cplx, noise_scale, t)
    assert torch.allclose(h_proj, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["srf", "srf_p", "pbg_only", "rcflow", "flower"])
@pytest.mark.parametrize("gm_output_mode", ["mean", "sample"])
def test_sample_channels_modes(mode, gm_output_mode):
    model = _tiny_model()
    _, p_cplx, y_cplx, noise_scale = _make_inputs()
    cfg = srf.SamplerConfig(
        mode=mode, gm_output_mode=gm_output_mode, nfe1=2, nfe2=2
    )
    h_est, diag = srf.sample_channels(model, y_cplx, p_cplx, noise_scale, cfg)

    assert h_est.shape == (BATCH, NR, NT)
    assert h_est.dtype == torch.complex64
    assert torch.isfinite(h_est).all()
    assert diag["sampler_mode"] == mode or mode == "flower"
    if mode == "rcflow":
        assert diag["model_output"] in ("gm", "velocity")
        return
    if mode == "flower":
        assert diag["flower_mode"] == gm_output_mode
        return
    assert diag["nfe1"] == 2 and diag["nfe2"] == 2
    for key in (
        "entropy_mean",
        "max_weight_mean",
        "logstd_mean",
        "tau2_learned_mean",
        "rc_var_mean",
        "pbg_residual_mean",
        "pbg_grad_norm_mean",
        "pbg_step_mean",
        "pbg_lipschitz_mean",
    ):
        assert key in diag
        assert math.isfinite(float(diag[key]))


def test_sampler_config_aliases_and_validation():
    assert srf.SamplerConfig(mode="baseline").mode == "srf_p"
    assert srf.SamplerConfig(mode="fig").mode == "pbg_only"
    assert srf.SamplerConfig(mode="fig_plus").mode == "srf"
    with pytest.raises(ValueError):
        srf.SamplerConfig(mode="unknown")
    with pytest.raises(ValueError):
        srf.SamplerConfig(gm_output_mode="unknown")


def test_flower_modes():
    model = _tiny_model()
    _, p_cplx, y_cplx, noise_scale = _make_inputs()

    for mode in ("mean", "sample"):
        h_est, diag = srf.flower_sample_channels(
            model, y_cplx, p_cplx, noise_scale, mode=mode, total_steps=2
        )
        assert h_est.shape == (BATCH, NR, NT)
        assert torch.isfinite(h_est).all()
        assert diag["flower_mode"] == mode
        assert diag["total_steps"] == 2

    h_mmse, diag_mmse = srf.flower_sample_channels(
        model,
        y_cplx,
        p_cplx,
        noise_scale,
        mode="mmse",
        total_steps=2,
        mmse_num_samples=2,
    )
    assert h_mmse.shape == (BATCH, NR, NT)
    assert torch.isfinite(h_mmse).all()
    assert diag_mmse["mmse_num_samples"] == 2


def test_training_objectives():
    model = _tiny_model()
    h_true, _, _, _ = _make_inputs()
    batch = {"H": h_true}

    for objective in (
        srf.GMFlowObjective(transition_ratio=0.5),
        srf.FlowMatchingObjective(),
    ):
        model.zero_grad(set_to_none=True)
        out = objective(model, batch)
        assert "loss" in out
        assert torch.isfinite(out["loss"])
        assert out["loss"].requires_grad
        out["loss"].backward()
        assert any(p.grad is not None for p in model.parameters())
        for key, value in out.items():
            if key != "loss":
                assert not value.requires_grad

    gm_objective = srf.GMFlowObjective()
    assert gm_objective.snr_list_db[0] == -10.0
    assert gm_objective.snr_list_db[-1] == 30.0
    times = gm_objective.sample_time(4, torch.device("cpu"))
    assert times.shape == (4,) and ((times > 0) & (times < 1)).all()


def test_get_dynamic_nfe():
    nfe1, nfe2 = srf.get_dynamic_nfe(torch.tensor([0.3]), 4)
    assert nfe1 == 100
    assert nfe2 >= 2


def test_checkpoint_helpers(tmp_path):
    model = _tiny_model()
    path = tmp_path / "tiny.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": {"tag": "tiny"}}, path)

    fresh = _tiny_model()
    missing, unexpected = srf.load_checkpoint(fresh, str(path), "cpu", strict=True)
    assert list(missing) == [] and list(unexpected) == []

    prefixed = {f"module.{key}": value for key, value in model.state_dict().items()}
    stripped = srf.strip_state_dict_prefix(prefixed)
    assert all(not key.startswith("module.") for key in stripped)


def test_mat_dataset_roundtrip(tmp_path):
    pytest.importorskip("hdf5storage")
    import numpy as np

    rng = np.random.default_rng(0)
    channels = (
        rng.standard_normal((NR, NT, 3)) + 1j * rng.standard_normal((NR, NT, 3))
    ).astype(np.complex64)
    path = tmp_path / "cdl_a.mat"
    srf.save_mat_dataset(path, channels, channels, channels)

    config = srf.ChannelConfig(image_size=(NR, NT), num_pilots=NPILOTS, snr_db=5.0)
    dataset = srf.MatChannelDataset([path], split="test", config=config, seed=0)
    assert len(dataset) == 3

    item = dataset[0]
    assert item["H"].shape == (NR, NT)
    assert item["P"].shape == (NT, NPILOTS)
    assert item["Y"].shape == (NR, NPILOTS)
    assert item["noise_scale"].shape == ()
    assert torch.isfinite(item["H"]).all()
