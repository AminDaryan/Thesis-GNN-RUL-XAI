## Set against: stored network, last cycle stage, best held-back pass restored

Stored run `fd001_physics_row_cap_5seed_varsplit_cfgba82e3_f4f40ad_20261002T152416Z`.

| Model | RMSE, 5 seeds | Network's RMSE minus this | Seeds network better | Paired t over seeds | 95 % interval over 20,000 resamplings of the test engines |
|---|---|---|---|---|---|
| stored network, last cycle stage, best held-back pass restored | 16.496 ± 0.415 | — | — | — | own RMSE [14.00, 18.90] |
| stored network, lstm stage, best held-back pass restored | 13.234 ± 0.723 | +3.262 | 0/5 | 0.0022 | [+1.15, +5.37] |
| constant: mean training label | 40.568 ± 0.089 | −24.072 | 5/5 | < 0.0001 | [−29.00, −19.12] |
| gradient boosting trained on shuffled labels, same 14 numbers | 40.846 ± 0.292 | −24.349 | 5/5 | < 0.0001 | [−29.33, −19.35] |
| gradient boosting on engine age alone | 31.239 ± 0.100 | −14.742 | 5/5 | < 0.0001 | [−18.92, −10.56] |
| linear regression, same 14 numbers | 20.457 ± 0.219 | −3.961 | 5/5 | < 0.0001 | [−6.30, −1.70] |
| one-hidden-layer network of 64, same 14 numbers | 17.213 ± 0.223 | −0.716 | 5/5 | 0.0230 | [−2.44, +1.11] |
| random forest, same 14 numbers | 18.062 ± 0.415 | −1.566 | 5/5 | 0.0026 | [−3.39, +0.42] |
| gradient boosting, same 14 numbers | 17.253 ± 0.329 | −0.756 | 5/5 | 0.0472 | [−2.56, +1.17] |
| gradient boosting, 42 window summaries | 12.140 ± 0.351 | +4.356 | 0/5 | 0.0001 | [+2.03, +6.68] |
| gradient boosting, window mean alone | 21.230 ± 0.506 | −4.733 | 5/5 | 0.0001 | [−8.65, −1.33] |
| gradient boosting, window slope alone | 13.693 ± 0.488 | +2.803 | 0/5 | 0.0002 | [−0.12, +5.68] |
| gradient boosting, last value and mean | 15.047 ± 0.631 | +1.449 | 0/5 | 0.0130 | [−0.39, +3.30] |
| gradient boosting, last value and slope | 11.835 ± 0.391 | +4.661 | 0/5 | 0.0002 | [+2.32, +6.97] |
| gradient boosting, mean and slope | 12.827 ± 0.320 | +3.669 | 0/5 | < 0.0001 | [+1.26, +6.10] |
| linear regression, 42 window summaries | 14.520 ± 0.120 | +1.976 | 0/5 | 0.0002 | [−0.34, +4.17] |

## Set against: stored network, lstm stage, best held-back pass restored

Stored run `fd001_physics_row_cap_5seed_varsplit_lstm32_cfgba82e3_19dceb5_20261002T210700Z`.

| Model | RMSE, 5 seeds | Network's RMSE minus this | Seeds network better | Paired t over seeds | 95 % interval over 20,000 resamplings of the test engines |
|---|---|---|---|---|---|
| stored network, last cycle stage, best held-back pass restored | 16.496 ± 0.415 | −3.262 | 5/5 | 0.0022 | [−5.37, −1.15] |
| stored network, lstm stage, best held-back pass restored | 13.234 ± 0.723 | — | — | — | own RMSE [11.20, 15.24] |
| constant: mean training label | 40.568 ± 0.089 | −27.334 | 5/5 | < 0.0001 | [−32.32, −22.29] |
| gradient boosting trained on shuffled labels, same 14 numbers | 40.846 ± 0.292 | −27.611 | 5/5 | < 0.0001 | [−32.64, −22.52] |
| gradient boosting on engine age alone | 31.239 ± 0.100 | −18.004 | 5/5 | < 0.0001 | [−22.26, −13.83] |
| linear regression, same 14 numbers | 20.457 ± 0.219 | −7.223 | 5/5 | < 0.0001 | [−10.12, −4.42] |
| one-hidden-layer network of 64, same 14 numbers | 17.213 ± 0.223 | −3.978 | 5/5 | 0.0001 | [−6.29, −1.68] |
| random forest, same 14 numbers | 18.062 ± 0.415 | −4.828 | 5/5 | 0.0002 | [−7.31, −2.30] |
| gradient boosting, same 14 numbers | 17.253 ± 0.329 | −4.018 | 5/5 | < 0.0001 | [−6.38, −1.64] |
| gradient boosting, 42 window summaries | 12.140 ± 0.351 | +1.094 | 0/5 | 0.0048 | [−0.10, +2.30] |
| gradient boosting, window mean alone | 21.230 ± 0.506 | −7.995 | 5/5 | < 0.0001 | [−12.35, −4.20] |
| gradient boosting, window slope alone | 13.693 ± 0.488 | −0.459 | 4/5 | 0.3515 | [−2.57, +1.65] |
| gradient boosting, last value and mean | 15.047 ± 0.631 | −1.813 | 5/5 | 0.0064 | [−3.56, −0.10] |
| gradient boosting, last value and slope | 11.835 ± 0.391 | +1.400 | 0/5 | 0.0037 | [+0.26, +2.55] |
| gradient boosting, mean and slope | 12.827 ± 0.320 | +0.407 | 1/5 | 0.2918 | [−0.98, +1.76] |
| linear regression, 42 window summaries | 14.520 ± 0.120 | −1.286 | 5/5 | 0.0178 | [−3.38, +0.58] |
