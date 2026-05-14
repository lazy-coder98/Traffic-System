"""
TrafficIQ — Offline Training Script

Trains all ML models and the Q-Learning RL agent, then persists them
to the models/ directory so the Flask app can load them instantly.

Usage:
    python train.py            # trains everything, saves to models/
    python train.py --force    # overwrites existing model files
"""

import os
import sys
import json
import time
import logging
import argparse

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import config
from rl_agent import QLearningTrafficManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────
def congestion_label(v: float) -> str:
    if v < config.LOW_CONGESTION_LIMIT:
        return "Low"
    elif v < config.HIGH_CONGESTION_LIMIT:
        return "Medium"
    return "High"


def load_and_prepare_data() -> pd.DataFrame:
    """Load the traffic CSV (or generate synthetic data) and feature-engineer."""
    if os.path.exists(config.DATA_PATH):
        logger.info("Loading dataset from %s", config.DATA_PATH)
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
                rows.append(
                    {"DateTime": ts, "Junction": junction, "Vehicles": vehicles}
                )
        df = pd.DataFrame(rows)

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.dayofweek

    # Cyclical encoding for hour & day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
    df["is_weekend"] = (df["day"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Encode junction labels
    le = LabelEncoder()
    df["Junction"] = le.fit_transform(df["Junction"].astype(str))

    # Congestion labels
    df["Congestion"] = df["Vehicles"].apply(congestion_label)

    logger.info(
        "Dataset ready: %s rows, %d junctions, %s → %s",
        f"{len(df):,}",
        df["Junction"].nunique(),
        str(df["DateTime"].min().date()),
        str(df["DateTime"].max().date()),
    )
    return df


def train_ml_models(df: pd.DataFrame, model_dir: str):
    """Train LR, RF regressor, and RF classifier; persist with joblib."""
    feature_cols = [
        "Junction", "hour", "day",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "is_weekend", "is_rush_hour",
    ]
    X = df[feature_cols]
    y = df["Vehicles"]
    y_class = df["Congestion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    _, _, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    # ── Linear Regression ─────────────────────────────────
    logger.info("Training Linear Regression…")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))
    lr_r2 = float(r2_score(y_test, lr_pred))
    logger.info("  LR  → RMSE=%.2f  R²=%.4f", lr_rmse, lr_r2)

    # ── Random Forest Regressor ───────────────────────────
    logger.info("Training Random Forest Regressor…")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))
    rf_r2 = float(r2_score(y_test, rf_pred))
    logger.info("  RF  → RMSE=%.2f  R²=%.4f", rf_rmse, rf_r2)

    # Cross-validation for RF
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1)
    logger.info("  RF CV R² = %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # ── Random Forest Classifier ─────────────────────────
    logger.info("Training Random Forest Classifier…")
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfc.fit(X_train, yc_train)
    rfc_acc = float(accuracy_score(yc_test, rfc.predict(X_test)))
    logger.info("  RFC → Accuracy=%.4f", rfc_acc)

    # ── Save models ──────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(lr, os.path.join(model_dir, "lr.joblib"))
    joblib.dump(rf, os.path.join(model_dir, "rf.joblib"))
    joblib.dump(rfc, os.path.join(model_dir, "rfc.joblib"))

    # ── Save metrics ─────────────────────────────────────
    metrics = {
        "lr": {"rmse": round(lr_rmse, 2), "r2": round(lr_r2, 4)},
        "rf": {
            "rmse": round(rf_rmse, 2),
            "r2": round(rf_r2, 4),
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
        },
        "rfc_accuracy": round(rfc_acc, 4),
        "feature_columns": feature_cols,
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("ML models saved to %s", model_dir)
    return metrics


def train_rl_agent(df: pd.DataFrame, model_dir: str):
    """Train Q-Learning RL agent and save Q-table + metadata."""
    logger.info(
        "Training RL agent (episodes=%d, α=%.2f, γ=%.2f, ε=%.2f)…",
        config.RL_EPISODES,
        config.RL_ALPHA,
        config.RL_GAMMA,
        config.RL_EPSILON,
    )

    agent = QLearningTrafficManager(
        n_junctions=config.RL_N_JUNCTIONS,
        alpha=config.RL_ALPHA,
        gamma=config.RL_GAMMA,
        epsilon=config.RL_EPSILON,
        episodes=config.RL_EPISODES,
    )
    agent.train(df)

    # Save Q-table
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, "q_table.npy"), agent.q_table)

    # Save metadata
    meta = {
        "reward_history": agent.reward_history,
        "converged_at": agent.converged_at,
        "total_reward": agent.total_reward,
    }
    with open(os.path.join(model_dir, "rl_metadata.json"), "w") as f:
        json.dump(meta, f)

    logger.info(
        "RL agent trained → converged at ep %d, total reward %.1f",
        agent.converged_at,
        agent.total_reward,
    )
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train TrafficIQ models")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model files",
    )
    args = parser.parse_args()

    model_dir = config.MODEL_DIR

    if not args.force and all(
        os.path.exists(os.path.join(model_dir, f))
        for f in ("rf.joblib", "rfc.joblib", "lr.joblib", "q_table.npy")
    ):
        logger.info("Models already exist. Use --force to retrain.")
        sys.exit(0)

    t0 = time.time()

    df = load_and_prepare_data()
    train_ml_models(df, model_dir)
    train_rl_agent(df, model_dir)

    elapsed = time.time() - t0
    logger.info("All training complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
