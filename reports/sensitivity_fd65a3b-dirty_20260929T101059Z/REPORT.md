# FD001 one-factor sensitivity study

This study changes one available choice at a time around a reference model and
repeats every configuration with seeds 42, 43, 44, 45 and 46. It is a result
about this complete reference configuration, not a universal ranking of the
choices.

The reference uses one long short-term memory network per sensor, the paper's
physics graph, row graph rescaling, graph layers without added offsets,
z-score sensor scaling, exponential smoothing, a one-step learning-rate drop,
20% of training engines held back, early stopping and capped test labels.

Lower is better for root mean squared error (RMSE), the NASA score and average
safe underestimation error (ASUE). Higher is better for R-squared. Every `±`
is the sample standard deviation across the five seeds. `ΔRMSE` is the value
for that row minus the reference value, paired by seed. A positive value is
worse. The p-value is the two-sided paired test over those five differences;
it is exploratory because seventeen comparisons were made.

| Changed factor | Level | RMSE | ΔRMSE | paired p | NASA score | R-squared | ASUE |
|---|---|---:|---:|---:|---:|---:|---:|
| Reference | Paper LSTM and physics graph | 13.234 ± 0.723 | 0.000 | 1.0000 | 316.3 ± 57.2 | 0.891 ± 0.012 | 8.146 ± 0.478 |
| Temporal stage | Last reading | 16.690 ± 0.229 | +3.456 | 0.0010 | 572.6 ± 35.9 | 0.827 ± 0.005 | 11.296 ± 1.133 |
| Temporal stage | Window average | 22.434 ± 0.516 | +9.200 | 0.000032 | 9973.3 ± 2462.4 | 0.686 ± 0.014 | 11.386 ± 1.232 |
| Temporal stage | GRU | 12.818 ± 0.681 | -0.416 | 0.4554 | 309.4 ± 62.6 | 0.897 ± 0.011 | 8.310 ± 0.894 |
| Temporal weights | One LSTM shared by all sensors | 13.569 ± 0.525 | +0.334 | 0.5222 | 398.3 ± 128.2 | 0.885 ± 0.009 | 8.459 ± 1.570 |
| Sensor graph | Every sensor pair | 13.636 ± 0.601 | +0.402 | 0.3723 | 330.8 ± 44.7 | 0.884 ± 0.010 | 9.521 ± 1.497 |
| Sensor graph | 41 strongest training correlations | 13.237 ± 0.967 | +0.002 | 0.9947 | 308.8 ± 58.3 | 0.890 ± 0.016 | 9.025 ± 0.942 |
| Graph rescaling | Symmetric by both endpoint degrees | 12.903 ± 0.701 | -0.332 | 0.3065 | 273.6 ± 18.0 | 0.896 ± 0.011 | 9.088 ± 1.161 |
| Graph-layer offset | Included | 13.516 ± 0.613 | +0.282 | 0.6478 | 329.0 ± 68.8 | 0.886 ± 0.010 | 8.656 ± 0.564 |
| Smoothing | Off | 13.164 ± 0.471 | -0.070 | 0.8044 | 293.7 ± 14.7 | 0.892 ± 0.008 | 8.774 ± 1.241 |
| Sensor scaling | Training range per condition | 12.186 ± 1.462 | -1.048 | 0.3319 | 266.6 ± 85.9 | 0.906 ± 0.024 | 7.984 ± 0.816 |
| Sensor scaling | Global training range | 12.186 ± 1.462 | -1.048 | 0.3319 | 266.6 ± 85.9 | 0.906 ± 0.024 | 7.984 ± 0.816 |
| Learning-rate path | Exponential | 13.656 ± 0.172 | +0.422 | 0.2317 | 318.1 ± 29.7 | 0.884 ± 0.003 | 9.055 ± 0.395 |
| Learning-rate path | Cosine | 13.844 ± 0.482 | +0.610 | 0.1461 | 341.8 ± 20.9 | 0.881 ± 0.008 | 9.446 ± 0.890 |
| Learning-rate path | Reduce after held-back error stalls | 13.234 ± 0.723 | 0.000 | 1.0000 | 316.3 ± 57.2 | 0.891 ± 0.012 | 8.146 ± 0.478 |
| Learning-rate path | Constant at the first rate | 13.234 ± 0.723 | 0.000 | 1.0000 | 316.3 ± 57.2 | 0.891 ± 0.012 | 8.146 ± 0.478 |
| Learning-rate scale | One tenth of the paper's rates | 13.663 ± 0.676 | +0.428 | 0.2740 | 345.0 ± 57.5 | 0.884 ± 0.012 | 8.895 ± 0.908 |
| Learning-rate scale | Ten times the paper's rates | 14.607 ± 0.307 | +1.372 | 0.0094 | 404.4 ± 67.6 | 0.867 ± 0.006 | 9.815 ± 1.973 |

## What the table establishes

- The recurrent temporal component matters under this configuration. Keeping
  only the final reading raised RMSE by 3.456 cycles, and replacing the sensor
  history with its average raised it by 9.200 cycles. Both differences remain
  below a conservative Bonferroni threshold of `0.05 / 17 = 0.00294`.
- A GRU scored 0.416 cycles lower than the LSTM, but the difference is small
  relative to the seed-to-seed changes and cannot be distinguished from zero
  with these five paired seeds.
- The physics and correlation graphs produced effectively the same mean RMSE:
  a difference of 0.002 cycles. The complete graph was 0.402 cycles worse, but
  that difference also cannot be distinguished from zero here.
- Turning smoothing off changed mean RMSE by only -0.070 cycles. This does not
  establish that smoothing never matters; it establishes that it did not move
  the five-seed mean detectably in this reference architecture.
- Per-condition and global range scaling are exactly the same on FD001 because
  FD001 has one operating condition. They are two names for the same arithmetic
  on this dataset, not independent evidence.
- The step, plateau and constant learning-rate paths produced identical model
  results. Every reference run selected a pass between 5 and 24 and stopped by
  pass 44, before the step schedule's halfway drop at pass 100. The plateau
  schedule also made no reduction before training stopped. Exponential and
  cosine schedules start changing earlier and therefore produced different
  results.
- Multiplying both learning rates by ten raised mean RMSE by 1.372 cycles. Its
  unadjusted paired p-value is 0.0094, but it does not pass the conservative
  correction for all seventeen comparisons. Dividing both rates by ten did not
  produce a distinguishable change.

## Test-label scoring choice

The reference predictions were also scored against uncapped test answers; the
network was not trained again. Uncapped scoring gave RMSE **14.398 ± 0.732**,
which is **1.164 cycles higher** than capped scoring (paired p = 0.000027).
It also gave NASA score 348.3 ± 60.1, R-squared 0.880 ± 0.012 and ASUE
10.473 ± 0.517. This is a change to the answers used for scoring, not an effect
on what the trained model predicts.

## Values deliberately held fixed

Graph widths `[32, 64, 32]`; first fully connected width `128`; 32 windows per
update; at most 200 passes; first seed 42; recurrent hidden size 32; one number
per sensor handed to the graph; label ceiling 125; window length 30; one cycle
between windows; smoothing weight 0.5 whenever smoothing is enabled.

## Reproducibility record

The 18 run folders contain 90 trained networks, test predictions, training
histories, engine splits and configuration files. Every run records commit
`fd65a3b-dirty`, the same three FD001 file hashes, and PyTorch 2.5.1 with CUDA
12.1. The exact commands are in `commands.csv`; `seed_results.csv` contains all
per-seed values; and `summary.csv` contains the unrounded means, standard
deviations, paired changes and paired p-values for all four measures.
