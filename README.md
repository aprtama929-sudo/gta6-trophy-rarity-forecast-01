# GTA VI Trophy Rarity Forecaster

![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aprtama929-sudo/gta6-trophy-rarity-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aprtama929-sudo/gta6-trophy-rarity-forecast-01/reward.svg)

An entry for the make.fun bounty [Forecast GTA VI trophy rarity week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

`main.py` (Python 3.13, standard library only) reads one case file — every read the house made of Exophase's
GTA VI PS5 trophy page before the checkpoint — and writes five quantiles (5%, 25%, 50%, 75%, 95%) of each
trophy's earn rate at the case's target, scored by pinball loss on the logit scale.

Method: per trophy, five median models (persistence; a logit-on-log-time trend; the same trend damped; a
logit-on-time trend; an exponential saturation curve toward a fitted ceiling) are blended with weights from
an internal backtest that cuts the series at several earlier times and scores each model against the reads
that followed. The spread comes from the same backtests, pooled across trophies for stability, plus a
model-disagreement term, widened with the square root of the forecast horizon.

Run it the way the grader does:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
