# TrafficIQ — Smart Congestion Prediction & Management Dashboard

A full-stack ML web app for predicting **and managing** urban traffic congestion using the Kaggle Traffic Prediction Dataset. Built with Flask, scikit-learn, Q-Learning RL, and a vanilla JS dashboard — all containerized with Docker.

## Architecture

```
User → Nginx (port 8080) → Static Dashboard
              ↓
         Flask API (port 5000) → Analysis Models (LR + Random Forest)
                                → Management Model (Q-Learning RL Agent)
                                         ↓
                                   traffic.csv (dataset)
```

**Analysis Pipeline:** Random Forest predicts vehicle counts and congestion levels.
**Management Pipeline:** Q-Learning RL agent learns optimal signal timing policies from historical data, replacing hard-coded rules with data-driven decisions.

## Quick Start

### Prerequisites
- Docker Desktop installed and running

### Run with Docker Compose
```bash
# Clone / unzip the project
cd traffic-ui

# Build and start both services
docker-compose up --build

# Then open your browser:
# Dashboard → http://localhost:8080
# API       → http://localhost:5000/api/stats
```

### Stop
```bash
docker-compose down
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/stats` | Dataset stats + model metrics |
| POST | `/api/predict` | Predict vehicles, congestion, signal time + RL recommendation |
| GET | `/api/heatmap` | Hour × Day traffic matrix |
| GET | `/api/junction_trend?junction=1` | Hourly avg per junction |
| GET | `/api/daily_pattern` | Daily avg across all junctions |
| GET | `/api/management/status` | RL policy summary for all junctions (current time) |
| POST | `/api/management/recommend` | RL recommendation for specific junction/hour/day |
| GET | `/api/management/policy_heatmap?junction=1` | RL signal durations as hour×day heatmap |
| GET | `/api/management/comparison` | RL vs rule-based comparison across all scenarios |

### POST /api/predict — Example
```json
// Request
{ "junction": 2, "hour": 18, "day": 4 }

// Response
{
  "predicted_vehicles": 47.3,
  "congestion_level": "Medium",
  "signal_time_seconds": 40,
  "rl_recommendation": {
    "optimal_signal_seconds": 50,
    "management_action": "Extend",
    "action_label": "Extend green phase",
    "priority_score": 45.2,
    "phase_allocation": { "green_pct": 45.5, "amber_seconds": 4, "red_seconds": 56.0, "cycle_seconds": 110.0 }
  }
}
```

## Dataset
- **File**: `traffic.csv` (place inside `app/`)
- **Source**: Kaggle Traffic Prediction Dataset
- **Records**: 48,120 hourly observations
- **Features**: DateTime, Junction (1–4), Vehicles

## Models

### Analysis (Traffic Prediction)
- **Linear Regression** — baseline
- **Random Forest Regressor** — production (better RMSE & R²)
- **Random Forest Classifier** — congestion level (Low / Medium / High)

### Management (Adaptive Signal Control)
- **Q-Learning RL Agent** — learns optimal signal timing policies
  - **96 discrete states**: junction × hour_bucket × day_type × congestion_level
  - **5 actions**: signal durations [15s, 25s, 35s, 50s, 70s]
  - **800 training episodes** with epsilon-greedy exploration
  - Outputs: optimal signal duration, phase allocation, priority score, management action

