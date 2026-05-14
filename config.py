"""
TrafficIQ — Centralised configuration.

All tuneable values live here and can be overridden via environment
variables or a .env file loaded by python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present


def _int(key, default):
    return int(os.getenv(key, default))


def _float(key, default):
    return float(os.getenv(key, default))


# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "traffic.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Congestion thresholds ────────────────────────────────────
LOW_CONGESTION_LIMIT = _int("CONGESTION_LOW", 35)
HIGH_CONGESTION_LIMIT = _int("CONGESTION_HIGH", 55)

# ── RL hyper-parameters ─────────────────────────────────────
RL_EPISODES = _int("RL_EPISODES", 400)
RL_ALPHA = _float("RL_ALPHA", 0.1)
RL_GAMMA = _float("RL_GAMMA", 0.95)
RL_EPSILON = _float("RL_EPSILON", 0.15)
RL_N_JUNCTIONS = _int("RL_N_JUNCTIONS", 4)

# ── Server ───────────────────────────────────────────────────
PORT = _int("PORT", 5000)
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

# ── Caching ──────────────────────────────────────────────────
CACHE_TTL_SECONDS = _int("CACHE_TTL", 300)  # 5 minutes
