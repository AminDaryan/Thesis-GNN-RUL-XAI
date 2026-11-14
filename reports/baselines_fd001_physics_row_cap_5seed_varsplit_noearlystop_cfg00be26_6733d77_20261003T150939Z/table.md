## Set against: stored network, last cycle stage, weights of the last pass

Stored run `fd001_physics_row_cap_5seed_varsplit_noearlystop_cfg00be26_19dceb5_20261002T183904Z`.

| Model | RMSE, 5 seeds | Network's RMSE minus this | Seeds network better | Paired t over seeds | 95 % interval over 20,000 resamplings of the test engines |
|---|---|---|---|---|---|
| stored network, last cycle stage, weights of the last pass | 16.818 ± 0.157 | — | — | — | own RMSE [14.14, 19.37] |
| stored network, lstm stage, weights of the last pass | 16.052 ± 0.975 | +0.766 | 1/5 | 0.1172 | [−2.43, +4.00] |
| constant: mean training label | 40.548 ± 0.000 | −23.730 | 5/5 | < 0.0001 | [−28.69, −18.73] |
| gradient boosting trained on shuffled labels, same 14 numbers | 40.546 ± 0.311 | −23.728 | 5/5 | < 0.0001 | [−28.71, −18.71] |
| gradient boosting on engine age alone | 31.180 ± 0.000 | −14.362 | 5/5 | < 0.0001 | [−18.56, −10.13] |
| linear regression, same 14 numbers | 20.394 ± 0.000 | −3.577 | 5/5 | < 0.0001 | [−5.92, −1.33] |
| one-hidden-layer network of 64, same 14 numbers | 17.100 ± 0.093 | −0.282 | 5/5 | 0.0454 | [−2.06, +1.58] |
| random forest, same 14 numbers | 17.954 ± 0.428 | −1.136 | 5/5 | 0.0009 | [−2.97, +0.80] |
| gradient boosting, same 14 numbers | 16.936 ± 0.000 | −0.118 | 4/5 | 0.1693 | [−1.93, +1.84] |
| gradient boosting, 42 window summaries | 12.470 ± 0.000 | +4.348 | 0/5 | < 0.0001 | [+1.78, +6.95] |
| gradient boosting, window mean alone | 21.521 ± 0.000 | −4.703 | 5/5 | < 0.0001 | [−9.31, −0.85] |
| gradient boosting, window slope alone | 13.531 ± 0.000 | +3.287 | 0/5 | < 0.0001 | [+0.08, +6.40] |
| gradient boosting, last value and mean | 14.341 ± 0.000 | +2.477 | 0/5 | < 0.0001 | [+0.40, +4.61] |
| gradient boosting, last value and slope | 12.138 ± 0.000 | +4.680 | 0/5 | < 0.0001 | [+2.15, +7.19] |
| gradient boosting, mean and slope | 12.333 ± 0.000 | +4.485 | 0/5 | < 0.0001 | [+1.92, +7.13] |
| linear regression, 42 window summaries | 14.439 ± 0.000 | +2.379 | 0/5 | < 0.0001 | [+0.02, +4.65] |

## Set against: stored network, lstm stage, weights of the last pass

Stored run `fd001_physics_row_cap_5seed_varsplit_lstm32_noearlystop_cfg00be26_19dceb5_20261003T144539Z`.

| Model | RMSE, 5 seeds | Network's RMSE minus this | Seeds network better | Paired t over seeds | 95 % interval over 20,000 resamplings of the test engines |
|---|---|---|---|---|---|
| stored network, last cycle stage, weights of the last pass | 16.818 ± 0.157 | −0.766 | 4/5 | 0.1172 | [−4.00, +2.43] |
| stored network, lstm stage, weights of the last pass | 16.052 ± 0.975 | — | — | — | own RMSE [13.68, 18.27] |
| constant: mean training label | 40.548 ± 0.000 | −24.496 | 5/5 | < 0.0001 | [−29.71, −19.36] |
| gradient boosting trained on shuffled labels, same 14 numbers | 40.546 ± 0.311 | −24.494 | 5/5 | < 0.0001 | [−29.71, −19.33] |
| gradient boosting on engine age alone | 31.180 ± 0.000 | −15.128 | 5/5 | < 0.0001 | [−19.63, −10.81] |
| linear regression, same 14 numbers | 20.394 ± 0.000 | −4.342 | 5/5 | 0.0006 | [−7.81, −1.02] |
| one-hidden-layer network of 64, same 14 numbers | 17.100 ± 0.093 | −1.048 | 4/5 | 0.0796 | [−4.36, +2.31] |
| random forest, same 14 numbers | 17.954 ± 0.428 | −1.902 | 5/5 | 0.0049 | [−5.34, +1.55] |
| gradient boosting, same 14 numbers | 16.936 ± 0.000 | −0.884 | 4/5 | 0.1128 | [−4.29, +2.56] |
| gradient boosting, 42 window summaries | 12.470 ± 0.000 | +3.583 | 0/5 | 0.0012 | [+1.56, +5.67] |
| gradient boosting, window mean alone | 21.521 ± 0.000 | −5.468 | 5/5 | 0.0002 | [−10.89, −0.72] |
| gradient boosting, window slope alone | 13.531 ± 0.000 | +2.521 | 0/5 | 0.0045 | [+0.01, +5.00] |
| gradient boosting, last value and mean | 14.341 ± 0.000 | +1.712 | 0/5 | 0.0172 | [−0.96, +4.32] |
| gradient boosting, last value and slope | 12.138 ± 0.000 | +3.914 | 0/5 | 0.0009 | [+1.93, +5.95] |
| gradient boosting, mean and slope | 12.333 ± 0.000 | +3.719 | 0/5 | 0.0010 | [+1.78, +5.64] |
| linear regression, 42 window summaries | 14.439 ± 0.000 | +1.614 | 0/5 | 0.0209 | [−1.03, +4.07] |
