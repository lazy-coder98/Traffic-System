"""
TrafficIQ — Flask API Server

Loads pre-trained ML models and RL agent from disk for instant startup.
Run `python train.py` first to generate the model files.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
import time
import logging
import warnings

import config
from rl_agent import (
    QLearningTrafficManager,
    signal_time,
    congestion_label,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Load & preprocess data ───────────────────────────────────
logger.info("Loading dataset…")
if os.path.exists(config.DATA_PATH):
    df = pd.read_csv(config.DATA_PATH)
else:
    logger.warning("Dataset not found — generating synthetic data")
    rng = np.random.default_rng(42)
    rows = []
    timestamps = pd.date_range("2015-01-01", periods=24 * 28, freq="h")
    for junction in range(1, 5):
        for ts in timestamps:
            rush_hour = 18 if junction in (1, 2) else 9
            base = 15 + junction * 4 + 12 * np.sin(
                (ts.hour - rush_hour) / 24 * 2 * np.pi
            )
            weekday_boost = 8 if ts.dayofweek < 5 else -2
            vehicles = max(2, int(base + weekday_boost + rng.normal(0, 5)))
            rows.append({"DateTime": ts, "Junction": junction, "Vehicles": vehicles})
    df = pd.DataFrame(rows)

df["DateTime"] = pd.to_datetime(df["DateTime"])
df["hour"] = df["DateTime"].dt.hour
df["day"] = df["DateTime"].dt.dayofweek

# Cyclical & interaction features (must match train.py)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
df["is_weekend"] = (df["day"] >= 5).astype(int)
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Junction"] = le.fit_transform(df["Junction"].astype(str))
df["Congestion"] = df["Vehicles"].apply(congestion_label)

FEATURE_COLS = [
    "Junction", "hour", "day",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_weekend", "is_rush_hour",
]

logger.info("Dataset loaded: %s rows", f"{len(df):,}")


# ── Load pre-trained models (or fallback to inline training) ─
def _build_feature_row(junction: int, hour: int, day: int) -> list:
    """Build a single feature row matching FEATURE_COLS order."""
    return [
        junction,
        hour,
        day,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day / 7),
        np.cos(2 * np.pi * day / 7),
        1 if day >= 5 else 0,
        1 if hour in (7, 8, 9, 17, 18, 19) else 0,
    ]


model_dir = config.MODEL_DIR
_model_files = ["lr.joblib", "rf.joblib", "rfc.joblib", "q_table.npy"]

if all(os.path.exists(os.path.join(model_dir, f)) for f in _model_files):
    logger.info("Loading pre-trained models from %s", model_dir)
    lr = joblib.load(os.path.join(model_dir, "lr.joblib"))
    rf = joblib.load(os.path.join(model_dir, "rf.joblib"))
    rfc = joblib.load(os.path.join(model_dir, "rfc.joblib"))

    # Load metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            model_metrics = json.load(f)
    else:
        model_metrics = {
            "lr": {"rmse": 0, "r2": 0},
            "rf": {"rmse": 0, "r2": 0},
            "rfc_accuracy": 0,
        }

    # Load RL agent
    rl_manager = QLearningTrafficManager(
        n_junctions=config.RL_N_JUNCTIONS,
        alpha=config.RL_ALPHA,
        gamma=config.RL_GAMMA,
        epsilon=config.RL_EPSILON,
        episodes=config.RL_EPISODES,
    )
    rl_manager.q_table = np.load(os.path.join(model_dir, "q_table.npy"))

    rl_meta_path = os.path.join(model_dir, "rl_metadata.json")
    if os.path.exists(rl_meta_path):
        with open(rl_meta_path) as f:
            rl_meta = json.load(f)
        rl_manager.reward_history = rl_meta.get("reward_history", [])
        rl_manager.converged_at = rl_meta.get("converged_at", 0)
        rl_manager.total_reward = rl_meta.get("total_reward", 0.0)

    logger.info("All models loaded from disk ✓")

else:
    logger.warning(
        "Pre-trained models not found — training inline (run `python train.py` to avoid this)"
    )
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

    X = df[FEATURE_COLS]
    y = df["Vehicles"]
    y_class = df["Congestion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    _, _, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, yc_train)

    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    model_metrics = {
        "lr": {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, lr_pred))), 2),
            "r2": round(float(r2_score(y_test, lr_pred)), 4),
        },
        "rf": {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, rf_pred))), 2),
            "r2": round(float(r2_score(y_test, rf_pred)), 4),
        },
        "rfc_accuracy": round(float(accuracy_score(yc_test, rfc.predict(X_test))), 4),
    }

    rl_manager = QLearningTrafficManager(
        n_junctions=4,
        alpha=config.RL_ALPHA,
        gamma=config.RL_GAMMA,
        epsilon=config.RL_EPSILON,
        episodes=config.RL_EPISODES,
    )
    rl_manager.train(df)
    logger.info("Inline training complete")


# ── Simple cache ─────────────────────────────────────────────
_cache: dict[str, dict] = {}


def cached(key: str, ttl: int = config.CACHE_TTL_SECONDS):
    """Decorator: cache JSON-serialisable return value for *ttl* seconds."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            now = time.time()
            entry = _cache.get(key)
            if entry and now < entry["expires"]:
                return entry["data"]
            result = fn(*args, **kwargs)
            _cache[key] = {"data": result, "expires": now + ttl}
            return result
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ── Validation helpers ───────────────────────────────────────
def _parse_junction(raw, one_indexed: bool = True) -> int:
    """Parse and validate a junction value (returns 0-indexed)."""
    try:
        j = int(raw)
    except (TypeError, ValueError):
        raise ValueError("Junction must be an integer (1-4)")
    if j < 1 or j > 4:
        raise ValueError("Junction must be between 1 and 4")
    return j - 1 if one_indexed else j


def _parse_hour(raw) -> int:
    try:
        h = int(raw)
    except (TypeError, ValueError):
        raise ValueError("Hour must be an integer (0-23)")
    if h < 0 or h > 23:
        raise ValueError("Hour must be between 0 and 23")
    return h


def _parse_day(raw) -> int:
    try:
        d = int(raw)
    except (TypeError, ValueError):
        raise ValueError("Day must be an integer (0-6)")
    if d < 0 or d > 6:
        raise ValueError("Day must be between 0 and 6")
    return d


def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), path)


@app.route("/api/stats")
def stats():
    return jsonify(
        {
            "total_records": int(len(df)),
            "junctions": 4,
            "date_range": {
                "start": str(df["DateTime"].min().date()),
                "end": str(df["DateTime"].max().date()),
            },
            "avg_vehicles": round(float(df["Vehicles"].mean()), 2),
            "max_vehicles": int(df["Vehicles"].max()),
            "congestion_dist": df["Congestion"].value_counts().to_dict(),
            "model_metrics": model_metrics,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return _error("Request body must be valid JSON")

    try:
        junction = _parse_junction(data.get("junction", 1))
        hour = _parse_hour(data.get("hour", 12))
        day = _parse_day(data.get("day", 0))
    except ValueError as e:
        return _error(str(e))

    sample = [_build_feature_row(junction, hour, day)]
    pred_vehicles = float(rf.predict(sample)[0])
    pred_congestion = congestion_label(pred_vehicles)
    pred_signal = signal_time(pred_vehicles)

    # RL management recommendation
    rl_rec = rl_manager.get_recommendation(junction, hour, day, pred_vehicles)

    logger.info(
        "Prediction: J%d H%d D%d → %.1f veh (%s) signal=%ds rl=%ds",
        junction + 1, hour, day, pred_vehicles, pred_congestion,
        pred_signal, rl_rec["optimal_signal_seconds"],
    )

    return jsonify(
        {
            "predicted_vehicles": round(pred_vehicles, 1),
            "congestion_level": pred_congestion,
            "signal_time_seconds": pred_signal,
            "rl_recommendation": rl_rec,
        }
    )


@app.route("/api/heatmap")
def heatmap():
    junction = request.args.get("junction")
    heatmap_df = df
    if junction is not None:
        try:
            j = _parse_junction(junction)
            heatmap_df = df[df["Junction"] == j]
        except ValueError as e:
            return _error(str(e))

    pivot = heatmap_df.pivot_table(
        values="Vehicles", index="hour", columns="day", aggfunc="mean"
    )
    result = []
    for hour in range(24):
        for day in range(7):
            val = pivot.get(day, {}).get(hour, 0)
            vehicles = round(float(val) if val else 0, 1)
            result.append(
                {
                    "hour": hour,
                    "day": int(day),
                    "vehicles": vehicles,
                    "congestion_level": congestion_label(vehicles),
                }
            )
    return jsonify(result)


@app.route("/api/junction_trend")
def junction_trend():
    try:
        junction = _parse_junction(request.args.get("junction", 1))
    except ValueError as e:
        return _error(str(e))

    sub = df[df["Junction"] == junction]
    hourly = sub.groupby("hour")["Vehicles"].mean().reset_index()
    return jsonify(
        [
            {"hour": int(r["hour"]), "avg_vehicles": round(float(r["Vehicles"]), 1)}
            for _, r in hourly.iterrows()
        ]
    )


@app.route("/api/daily_pattern")
def daily_pattern():
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("day")["Vehicles"].mean().reset_index()
    return jsonify(
        [
            {"day": days[int(r["day"])], "avg_vehicles": round(float(r["Vehicles"]), 1)}
            for _, r in daily.iterrows()
        ]
    )


# ── Management API routes ────────────────────────────────────
@app.route("/api/management/status")
def management_status():
    """Current RL policy summary for all junctions at the current time."""
    from datetime import datetime

    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()

    junction_status = []
    for j in range(4):
        sample = [_build_feature_row(j, current_hour, current_day)]
        pred_v = float(rf.predict(sample)[0])
        rec = rl_manager.get_recommendation(j, current_hour, current_day, pred_v)
        junction_status.append(
            {"junction": j + 1, "predicted_vehicles": round(pred_v, 1), **rec}
        )

    # Sort by priority (highest first)
    junction_status.sort(key=lambda x: x["priority_score"], reverse=True)

    return jsonify(
        {
            "timestamp_hour": current_hour,
            "timestamp_day": current_day,
            "junctions": junction_status,
            "training_metrics": {
                "episodes": rl_manager.episodes,
                "converged_at": rl_manager.converged_at,
                "total_reward": round(rl_manager.total_reward, 1),
                "avg_reward_last_50": (
                    round(float(np.mean(rl_manager.reward_history[-50:])), 1)
                    if rl_manager.reward_history
                    else 0
                ),
                "reward_history_sample": [
                    round(float(r), 1)
                    for r in rl_manager.reward_history[
                        :: max(1, len(rl_manager.reward_history) // 40)
                    ]
                ],
            },
        }
    )


@app.route("/api/management/recommend", methods=["POST"])
def management_recommend():
    """Get RL recommendation for a specific junction/hour/day."""
    data = request.get_json(silent=True)
    if not data:
        return _error("Request body must be valid JSON")

    try:
        junction = _parse_junction(data.get("junction", 1))
        hour = _parse_hour(data.get("hour", 12))
        day = _parse_day(data.get("day", 0))
    except ValueError as e:
        return _error(str(e))

    sample = [_build_feature_row(junction, hour, day)]
    pred_v = float(rf.predict(sample)[0])
    rec = rl_manager.get_recommendation(junction, hour, day, pred_v)

    return jsonify(
        {
            "junction": junction + 1,
            "hour": hour,
            "day": day,
            "predicted_vehicles": round(pred_v, 1),
            **rec,
        }
    )


@app.route("/api/management/policy_heatmap")
@cached("policy_heatmap")
def management_policy_heatmap():
    """Return the RL-recommended signal durations as a heatmap per junction."""
    try:
        junction = _parse_junction(request.args.get("junction", 1))
    except ValueError as e:
        return _error(str(e))

    # Batch predict all 168 combinations (24 hours × 7 days)
    scenarios = [_build_feature_row(junction, h, d) for h in range(24) for d in range(7)]
    all_preds = rf.predict(scenarios)

    result = []
    idx = 0
    for hour in range(24):
        for day in range(7):
            pred_v = float(all_preds[idx])
            rec = rl_manager.get_recommendation(junction, hour, day, pred_v)
            rule_sig = signal_time(pred_v)
            result.append(
                {
                    "hour": hour,
                    "day": day,
                    "rl_signal": rec["optimal_signal_seconds"],
                    "rule_signal": rule_sig,
                    "management_action": rec["management_action"],
                    "priority_score": rec["priority_score"],
                    "predicted_vehicles": round(pred_v, 1),
                }
            )
            idx += 1
    return jsonify(result)


@app.route("/api/management/comparison")
@cached("comparison")
def management_comparison():
    """Compare RL policy vs rule-based across all junctions and time periods."""
    # Pre-calculate all 672 predictions in one batch
    all_scenarios = [
        _build_feature_row(j, h, d) for j in range(4) for h in range(24) for d in range(7)
    ]
    all_preds = rf.predict(all_scenarios)

    # Map predictions back to (j, hour, day)
    pred_map = {}
    idx = 0
    for j in range(4):
        for hour in range(24):
            for day in range(7):
                pred_map[(j, hour, day)] = all_preds[idx]
                idx += 1

    comparison = []
    total_rl = 0
    total_rule = 0
    total_scenarios = 0

    for j in range(4):
        junction_data = {"junction": j + 1, "periods": []}
        for hb_idx, hb_name in enumerate(rl_manager.HOUR_BUCKET_NAMES):
            hb_lo, hb_hi = rl_manager.HOUR_BUCKETS[hb_idx]

            rl_signals = []
            rule_signals = []
            for hour in range(hb_lo, hb_hi):
                for day in range(7):
                    pred_v = float(pred_map[(j, hour, day)])
                    rec = rl_manager.get_recommendation(j, hour, day, pred_v)
                    rl_signals.append(rec["optimal_signal_seconds"])
                    rule_signals.append(signal_time(pred_v))

            avg_rl = round(float(np.mean(rl_signals)), 1)
            avg_rule = round(float(np.mean(rule_signals)), 1)
            total_rl += sum(rl_signals)
            total_rule += sum(rule_signals)
            total_scenarios += len(rl_signals)

            junction_data["periods"].append(
                {
                    "period": hb_name,
                    "avg_rl_signal": avg_rl,
                    "avg_rule_signal": avg_rule,
                    "difference": round(avg_rule - avg_rl, 1),
                }
            )
        comparison.append(junction_data)

    efficiency = (
        round((1 - total_rl / max(total_rule, 1)) * 100, 1) if total_rule > 0 else 0
    )

    return jsonify(
        {
            "comparison": comparison,
            "overall": {
                "avg_rl_signal": round(total_rl / max(total_scenarios, 1), 1),
                "avg_rule_signal": round(total_rule / max(total_scenarios, 1), 1),
                "signal_efficiency_pct": efficiency,
                "total_scenarios": total_scenarios,
            },
        }
    )


# ── Data export ──────────────────────────────────────────────
@app.route("/api/export/policy")
def export_policy():
    """Export RL policy as CSV for offline analysis."""
    summary = rl_manager.get_policy_summary()
    df_policy = pd.DataFrame(summary)
    return (
        df_policy.to_csv(index=False),
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": "attachment; filename=rl_policy.csv",
        },
    )


@app.route("/api/export/predictions")
def export_predictions():
    """Export all junction predictions for current conditions as CSV."""
    from datetime import datetime

    now = datetime.now()
    rows = []
    for j in range(4):
        for h in range(24):
            for d in range(7):
                sample = [_build_feature_row(j, h, d)]
                pred_v = float(rf.predict(sample)[0])
                rec = rl_manager.get_recommendation(j, h, d, pred_v)
                rows.append(
                    {
                        "junction": j + 1,
                        "hour": h,
                        "day": d,
                        "predicted_vehicles": round(pred_v, 1),
                        "congestion": congestion_label(pred_v),
                        "rl_signal_s": rec["optimal_signal_seconds"],
                        "rule_signal_s": signal_time(pred_v),
                        "management_action": rec["management_action"],
                        "priority_score": rec["priority_score"],
                    }
                )
    df_export = pd.DataFrame(rows)
    return (
        df_export.to_csv(index=False),
        200,
        {
            "Content-Type": "text/csv",
            "Content-Disposition": "attachment; filename=traffic_predictions.csv",
        },
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)
