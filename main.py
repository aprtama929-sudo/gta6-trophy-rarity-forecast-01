"""gta6-trophy-rarity-forecast-01: a growth-curve forecaster for GTA VI trophy earn rates.

The case gives every read the house made of Exophase's GTA VI PS5 trophy page before a checkpoint: each trophy's
earn rate (the percent of tracked players who have earned it) at each read. The entry forecasts each rate at the
case's target as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95), scored by pinball loss on the logit scale.

Method, per trophy, all on the logit scale unless noted:

  - Five median models are fitted to the trophy's series: persistence (the last rate), a least-squares line of
    logit(rate) on ln(1 + hours since launch), the same line with its slope damped, a least-squares line of
    logit(rate) on hours, and an exponential saturation curve in rate space (rate approaches a fitted ceiling
    C, C chosen from a grid by least squares on ln(C - rate) against time). The trend models are anchored at
    the last point, and extrapolations are capped so no model can run away.
  - The models are blended with weights from an internal backtest: the series is cut at several earlier times,
    each model is refitted on the prefix and scored against the reads that followed, and the blend favours the
    models that would have forecast this trophy best. A flat series makes every model agree, so the blend is
    persistence.
  - The spread comes from the same backtests: the blend's forecast errors per square-root day, pooled across
    trophies for stability, plus a term for how far the models disagree, widened like the square root of the
    horizon. A very short history gets a wide floor: on launch night a single Exophase refresh can rebase
    every rate at once.

Deterministic throughout (no randomness, no clock), standard library only. Run by the grader as
`python3 -P -s /bot/main.py` once per case; outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, in case a module ever ships beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The curve baseline's time origin: the game's release day at 2026-11-19T00:00:00Z.
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Fitting never uses more than this many points of a series (the tail is kept whole).
MAX_FIT_POINTS = 200
# How much of the blend's move away from the last rate the median keeps: trend models overshoot when a
# trophy's growth bends toward a ceiling the series has not shown yet, so the move is shrunk a little.
MEDIAN_SHRINK = 0.85


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment: datetime) -> float:
    return (moment - LAUNCH).total_seconds() / 3600.0


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares (intercept, slope); a degenerate x spread gives (mean(y), 0)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    if sxx <= 1e-12:
        return my, 0.0
    slope = sxy / sxx
    return my - slope * mx, slope


def thin(points: list[tuple[float, float]], cap: int) -> list[tuple[float, float]]:
    """Evenly thin a series to at most `cap` points, always keeping the first and last."""
    if len(points) <= cap:
        return points
    step = (len(points) - 1) / (cap - 1)
    kept = [points[round(i * step)] for i in range(cap)]
    kept[-1] = points[-1]
    return kept


# --- Median models: each takes the series [(hours since launch, rate)] and returns a function of t (hours) ---
# --- giving a logit, or None when the model cannot be fitted. Predictions are capped near the data.        ---


def model_persistence(points: list[tuple[float, float]]):
    last = logit(points[-1][1])
    return lambda t: last


def model_log_time(points: list[tuple[float, float]]):
    if len(points) < 3:
        return None
    xs = [math.log1p(max(t, 0.0)) for t, _ in points]
    ys = [logit(r) for _, r in points]
    a, b = linfit(xs, ys)
    b = min(3.0, max(-3.0, b))
    last_t, last_r = points[-1]
    last = logit(last_r)

    def predict(t: float) -> float:
        change = b * (math.log1p(max(t, 0.0)) - math.log1p(max(last_t, 0.0)))
        return last + min(4.0, max(-4.0, change))

    return predict


def model_log_time_damped(points: list[tuple[float, float]]):
    """The log-time trend with its slope damped: growth that is already slowing counts for less."""
    base = model_log_time(points)
    if base is None:
        return None
    last = logit(points[-1][1])
    t_last = points[-1][0]

    def predict(t: float) -> float:
        return last + 0.6 * (base(t) - last) if t > t_last else base(t)

    return predict


def model_linear_time(points: list[tuple[float, float]]):
    if len(points) < 3:
        return None
    xs = [t for t, _ in points]
    ys = [logit(r) for _, r in points]
    _, b = linfit(xs, ys)
    b = min(0.15, max(-0.05, b))
    last_t, last_r = points[-1]
    last = logit(last_r)

    def predict(t: float) -> float:
        change = b * (t - last_t)
        return last + min(2.5, max(-2.5, change))

    return predict


def model_saturation(points: list[tuple[float, float]]):
    """rate(t) = C - exp(a + b t) with b < 0: an exponential approach to a ceiling C, fitted by grid search."""
    if len(points) < 4:
        return None
    r_max = max(r for _, r in points)
    lo = max(r_max * 1.01, r_max + 0.02)
    if lo >= 99.99:
        return None
    best = None  # (sse, a, b, C)
    grid = 33
    for i in range(grid):
        ceiling = lo + (99.99 - lo) * (i / (grid - 1)) ** 1.5
        xs = []
        ys = []
        for t, r in points:
            if r < ceiling:
                xs.append(t)
                ys.append(math.log(ceiling - r))
        if len(xs) < 4:
            continue
        a, b = linfit(xs, ys)
        if b >= -1e-9:
            continue
        sse = 0.0
        for x, y in zip(xs, ys):
            d = a + b * x - y
            sse += d * d
        if best is None or sse < best[0]:
            best = (sse, a, b, ceiling)
    if best is None:
        return None
    _, a, b, ceiling = best
    last_t, last_r = points[-1]
    last = logit(last_r)

    def rate_at(t: float) -> float:
        exponent = a + b * t
        gap = math.exp(exponent) if exponent < 700 else float("inf")
        return ceiling - gap

    # Anchor at the last point in logit space: keep the curve's shape, drop its level error.
    shift = last - logit(max(rate_at(last_t), 0.01))

    def predict(t: float) -> float:
        rate = rate_at(t)
        raw = logit(rate) if rate > 0 else logit(0.01)
        return last + min(4.0, max(-4.0, raw + shift - last))

    return predict


MODEL_BUILDERS = [model_persistence, model_log_time, model_log_time_damped, model_linear_time, model_saturation]


def build_models(points: list[tuple[float, float]]):
    """Fit every model that can be fitted; returns a list of (builder index, predict fn)."""
    fitted = []
    for idx, builder in enumerate(MODEL_BUILDERS):
        try:
            fn = builder(points)
        except Exception:
            fn = None
        if fn is not None:
            fitted.append((idx, fn))
    return fitted


def backtest(points: list[tuple[float, float]]):
    """Cut the series at several earlier times, refit each model on the prefix, and score it against the
    reads that came after the cut, each error divided by the square root of its horizon in days.

    Returns (weights, sigma_per_sqrt_day or None): weights maps builder index to blend weight, from each
    model's mean normalized error; sigma is the blend's own mean normalized error, a per-square-root-day
    spread in logits. With too little history both are None and the caller falls back to persistence.
    """
    t_first = points[0][0]
    t_last = points[-1][0]
    span = t_last - t_first
    if span <= 0 or len(points) < 6:
        return None, None
    err_sum = {}
    err_n = {}
    for frac in (0.4, 0.55, 0.7, 0.85):
        cut = t_first + frac * span
        prefix = [p for p in points if p[0] <= cut]
        future = [p for p in points if p[0] > cut]
        if len(prefix) < 4 or not future:
            continue
        future = thin(future, 10)
        for idx, fn in build_models(thin(prefix, 120)):
            for t_read, r_read in future:
                horizon_days = max((t_read - cut) / 24.0, 0.02)
                err = abs(fn(t_read) - logit(r_read)) / math.sqrt(horizon_days)
                err_sum[idx] = err_sum.get(idx, 0.0) + err
                err_n[idx] = err_n.get(idx, 0) + 1
    if not err_n:
        return None, None
    mean_err = {idx: err_sum[idx] / err_n[idx] for idx in err_sum}
    raw = {idx: 1.0 / (mean_err[idx] + 0.15) ** 2 for idx in mean_err}
    total = sum(raw.values())
    weights = {idx: w / total for idx, w in raw.items()}
    sigma = sum(mean_err[idx] * weights[idx] for idx in mean_err)
    return weights, sigma


def forecast_trophy(points: list[tuple[float, float]], t_target: float, horizon_days: float,
                    pool_sigma: float | None) -> tuple[float, float]:
    """Returns (median logit, spread in logits) for one trophy's rate at t_target."""
    fitted = build_models(thin(points, MAX_FIT_POINTS))
    weights, sigma = backtest(points)
    last_logit = logit(points[-1][1])
    if weights is None:
        weights = {fitted[0][0]: 1.0} if fitted else {}
    usable = [(idx, fn) for idx, fn in fitted if idx in weights]
    if not usable:
        usable = fitted
        weights = {idx: 1.0 / len(usable) for idx, _ in usable} if usable else {}
    if not usable:
        return last_logit, max(0.35, 0.5 * math.sqrt(horizon_days))
    total_w = sum(weights[idx] for idx, _ in usable)
    preds = {idx: fn(t_target) for idx, fn in usable}
    median = sum(preds[idx] * weights[idx] for idx, _ in usable) / total_w
    median = last_logit + MEDIAN_SHRINK * (median - last_logit)
    # Model disagreement widens the spread.
    disagree = math.sqrt(sum(weights[idx] * (preds[idx] - median) ** 2 for idx, _ in usable) / total_w)
    # Per-trophy sigma, shrunk toward the cross-trophy pool, itself shrunk toward the random-walk 0.5.
    own = sigma if sigma is not None else 0.5
    if pool_sigma is not None:
        own = math.sqrt(0.6 * own * own + 0.4 * pool_sigma * pool_sigma)
    own = math.sqrt(0.5 * own * own + 0.5 * 0.25)
    spread = math.sqrt((own * math.sqrt(horizon_days)) ** 2 + (0.6 * disagree) ** 2)
    span_hours = points[-1][0] - points[0][0]
    floor = max(0.30, 0.32 * math.sqrt(horizon_days))
    if span_hours < 12.0:
        # A short, launch-night history can be rebased wholesale by one Exophase refresh.
        floor = max(floor, 1.0)
    spread = min(2.5, max(floor, spread))
    return median, spread


def series_for(case: dict, index: int) -> list[tuple[float, float]]:
    points = []
    for read in case["history"]:
        rate = read["rates"][index]
        if rate is None:
            continue
        moment = read.get("at") or read.get("slot")
        points.append((hours_since_launch(parse_time(moment)), float(rate)))
    points.sort(key=lambda p: p[0])
    return points


def forecast_case(case: dict, name: str) -> dict:
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    t_target = hours_since_launch(parse_time(case["target"]))
    horizon_days = max(case["horizon_minutes"] / 1440.0, 0.01)
    all_points = [series_for(case, i) for i in range(len(case["trophies"]))]
    # First pass: per-trophy backtest sigmas, pooled for stability.
    sigmas = []
    for points in all_points:
        if len(points) >= 6:
            _, sigma = backtest(points)
            if sigma is not None:
                sigmas.append(sigma)
    pool_sigma = None
    if sigmas:
        sigmas.sort()
        pool_sigma = sigmas[len(sigmas) // 2]
    # Fallback level for a trophy with no usable rate anywhere: the median last rate of the others.
    lasts = [pts[-1][1] for pts in all_points if pts]
    lasts.sort()
    fallback_rate = lasts[len(lasts) // 2] if lasts else 1.0
    rows = []
    for trophy, points in zip(case["trophies"], all_points):
        if points:
            median, spread = forecast_trophy(points, t_target, horizon_days, pool_sigma)
        else:
            median, spread = logit(fallback_rate), max(0.6, 0.8 * math.sqrt(horizon_days))
        quantiles = []
        previous = 0.0
        for z in Z:
            q = inv_logit(median + z * spread)
            q = min(100.0, max(0.0, q))
            q = max(q, previous)
            quantiles.append(round4(q))
            previous = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def persistence_case(case: dict) -> dict:
    """The house baseline's forecast, used only if the main forecaster ever fails on a case."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0)
        rows.append({"id": trophy["id"], "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = persistence_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
