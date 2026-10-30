# Paper results (reference values)

Numbers below are copied from the accepted paper (Table I) for reference. They
were obtained on 3GPP CDL-A/B/C/D channels with `N_r = 16`, `N_t = 64`,
`α = N_p/N_t = 0.6`, and 100 test realizations per profile; values are
`NMSE = 10 log10(||Ĥ-H||_F² / ||H||_F²)` in dB, averaged over CDL-A/B/C/D.

| Method | -10 | -5 | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| LMMSE | -1.41 | -2.80 | -4.69 | -6.83 | -9.09 | -11.23 | -13.17 | -14.67 | -15.62 |
| LASSO | 4.94 | 0.95 | -2.26 | -4.93 | -6.98 | -8.41 | -9.35 | -9.86 | -10.07 |
| fsAD | 11.17 | 6.40 | 2.01 | -1.84 | -4.89 | -6.86 | -7.88 | -8.33 | -8.42 |
| LDAMP | 11.80 | 5.28 | -1.33 | -8.47 | -14.21 | -16.07 | -16.78 | -17.05 | -17.12 |
| DMCE | -2.26 | -4.27 | -6.27 | -8.26 | -10.90 | -13.17 | -14.88 | -16.12 | -16.97 |
| RC-Flow | -0.74 | -4.26 | -7.62 | -10.68 | -13.75 | -16.67 | -19.49 | -22.04 | -24.22 |
| SRF-P (DPR only) | -0.60 | -4.06 | -7.52 | -10.80 | -13.97 | -16.98 | -19.78 | -22.43 | -24.70 |
| SRF (DPR + PBG) | -1.53 | -4.52 | -7.55 | -10.79 | -13.90 | -16.88 | -19.78 | -22.42 | -24.70 |

@ SNR: -10, -5, 0, 5, 10, 15, 20, 25, 30 dB.

Reported observations from the paper:

- SRF improves over RC-Flow at eight of nine SNR points (mean gain 0.29 dB);
  the only deficit is 0.07 dB at 0 dB.
- PBG gives its clearest benefit at low SNR (-10 and -5 dB); SRF-P is nearly
  identical or slightly better at medium-to-high SNR, so full SRF is mainly
  preferred in noise-limited regimes.

## Reproducing the table with this repository

1. Prepare the CDL data (see `docs/DATA.md`).
2. Train the SRF backbone (DPR objective, `r = 0.5`):

   ```bash
   python scripts/train.py --config configs/srf_cdl_16x64_alpha06.yaml
   ```

3. Evaluate both ablation points with the trained checkpoint:

   ```bash
   python scripts/evaluate.py \
       --config configs/eval_cdl_16x64_alpha06.yaml \
       --checkpoint outputs/srf_cdl_16x64_alpha06/best_model.pt \
       --samplers srf_p srf \
       --output-json results/table1_srf.json
   ```

   `srf_p` reproduces the "SRF-P" row (DPR without PBG) and `srf` the full
   "SRF" row. Classical baselines (LMMSE, LASSO, fsAD, LDAMP, DMCE) are not
   distributed with this release; the RC-Flow and FLOWER comparison samplers
   used in our implementation are available via
   `--samplers rcflow` and `--samplers flower` (they require the corresponding
   checkpoints).
