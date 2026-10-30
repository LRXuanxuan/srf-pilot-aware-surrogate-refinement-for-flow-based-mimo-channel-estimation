# Method: SRF, DPR and PBG

Summary of the method implemented in `src/srf`. Equation numbers refer to the
paper, "SRF: Pilot-Aware Surrogate Refinement for Flow-Based MIMO Channel
Estimation"; code names are listed in the table at the end.

## 1. Flow prior and DPR (distribution-aware prior refinement)
The observation is `Y = H P + N` (`H in C^{N_r x N_t}`, `P in C^{N_t x N_p}`,
`alpha = N_p / N_t` pilots, Gaussian noise of standard deviation `sigma`). A
U-Net predicts the mixture `p(u | H_t, t)` of the displacement `u` on the
additive path `H_t = H_0 + t u` (`t = 1` noise end, `t = 0` clean end); DPR uses
`K = 1`. Training minimises a finite-transition NLL: with `t_low = (1 - r) t_high`,
`r = 0.5`, the predicted mixture of `u` is mapped through the additive
transition and its NLL is evaluated at the realised `H_t_low`. At inference DPR
supplies the refined clean-channel centre (paper eq. (25))
```
H0_hat = H_t - t * mu_theta(H_t, t),
```
where `mu_theta` is the mixture mean; `gm_output_mode` selects the mean
(`"mean"`) or one mixture sample (`"sample"`, legacy default).

## 2. Pilot-consistency proximal update (SRF-P)
With `v_t^2 = t^2 / (t^2 + (1 - t)^2 + 1e-6)`, the DPR centre is coupled to the
pilot likelihood in closed form (paper eq. (27), `proximal_update`):
```
H_proj = (Y P^H / sigma^2 + H0_hat / (v_t^2 + eps))
         (P P^H / sigma^2 + I / (v_t^2 + eps))^{-1}.
```
DPR + eq. (27) is SRF-P (`mode="srf_p"`, legacy name `baseline`).

## 3. PBG (posterior bridge guidance)
After the reverse step re-embeds the trajectory as
`H_{t_next} = t_next * epsilon + (1 - t_next) * H_proj`, PBG corrects the
pilot-space residual (paper eqs. (29)-(36)); with `sigma_t = t_next`:
```
y_tilde = (1 - sigma_t) Y + w * sigma_t * P epsilon,
g       = (P H - y_tilde) P^H / L_op,      L_op = lambda_max(P P^H),
eta_t   = min( c * t_curr / (1 - t_curr + 1e-6) * alpha / (1 + sigma^2),
               step_max ),
H       <- H - eta_t * g,                  (K iterations)
```
with `w = 1`, `c = 0.10`, `step_max = 0.5`, `K = 1`. The legacy factor
`sigma_curr / alpha_curr` equals `t_curr / (1 - t_curr + 1e-6)` and the pilot
gate is `min(alpha, 1)`. Full SRF recombines null/range parts with `rho = 0.08`:
```
H_tilde = (1 - t_next) H0_hat + t_next * epsilon,
H <- (1 - rho) H + rho * ( (I - P P^dagger) H_tilde + P P^dagger H ),
```
i.e. paper Algorithm 1 (`mode="srf"`, legacy `fig_plus`); `mode="pbg_only"`
(legacy `fig`) applies PBG without the recombination.

## 4. Code names versus paper names
| Code | Paper |
| --- | --- |
| `mode="srf"` | full SRF (DPR + PBG + null/range mixing, Algorithm 1) |
| `mode="srf_p"` | SRF-P (DPR + eq. (27), no PBG) |
| `mode="pbg_only"` | PBG ablation without null/range mixing |
| `mode="rcflow"` / `srf.baselines.flower` | RC-Flow and FLOWER baselines |
| `pbg_*` settings | PBG (legacy name "FIG") |
| `lambda_exp`, `beta_exp` | time-warp exponents (`LAMBDA = BETA = 2`) |

`docs/NAMING.md` has the complete legacy-to-release mapping and the list of
intentionally dropped search/evaluation harnesses and hard-coded paths.
