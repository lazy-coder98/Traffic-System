# TrafficIQ — Smart Congestion Prediction Dashboard

A full-stack ML web app for predicting urban traffic congestion using the Kaggle Traffic Prediction Dataset. Built with Flask, scikit-learn, and a vanilla JS dashboard — all containerized with Docker.

## Architecture

```
User → Nginx (port 8080) → Static Dashboard
              ↓
         Flask API (port 5000) → ML Models (LR + Random Forest)
                                         ↓
                                   traffic.csv (dataset)
```

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
| POST | `/api/predict` | Predict vehicles, congestion, signal time |
| GET | `/api/heatmap` | Hour × Day traffic matrix |
| GET | `/api/junction_trend?junction=1` | Hourly avg per junction |
| GET | `/api/daily_pattern` | Daily avg across all junctions |

### POST /api/predict — Example
```json
// Request
{ "junction": 2, "hour": 18, "day": 4 }

// Response
{
  "predicted_vehicles": 47.3,
  "congestion_level": "Medium",
  "signal_time_seconds": 40
}
```

## Dataset
- **File**: `traffic.csv` (place inside `app/`)
- **Source**: Kaggle Traffic Prediction Dataset
- **Records**: 48,120 hourly observations
- **Features**: DateTime, Junction (1–4), Vehicles

## Models
- **Linear Regression** — baseline
- **Random Forest Regressor** — production (better RMSE & R²)
- **Random Forest Classifier** — congestion level (Low / Medium / High)
