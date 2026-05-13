from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import os

app = Flask(__name__)
CORS(app)

# ── Load & preprocess ──────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "traffic.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    rng = np.random.default_rng(42)
    rows = []
    timestamps = pd.date_range("2015-01-01", periods=24 * 28, freq="h")
    for junction in range(1, 5):
        for ts in timestamps:
            rush_hour = 18 if junction in (1, 2) else 9
            base = 15 + junction * 4 + 12 * np.sin((ts.hour - rush_hour) / 24 * 2 * np.pi)
            weekday_boost = 8 if ts.dayofweek < 5 else -2
            vehicles = max(2, int(base + weekday_boost + rng.normal(0, 5)))
            rows.append({"DateTime": ts, "Junction": junction, "Vehicles": vehicles})
    df = pd.DataFrame(rows)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['hour'] = df['DateTime'].dt.hour
df['day']  = df['DateTime'].dt.dayofweek

le = LabelEncoder()
df['Junction'] = le.fit_transform(df['Junction'].astype(str))

LOW_CONGESTION_LIMIT = 35
HIGH_CONGESTION_LIMIT = 55

def congestion(v):
    if v < LOW_CONGESTION_LIMIT:  return "Low"
    elif v < HIGH_CONGESTION_LIMIT: return "Medium"
    else:        return "High"

def signal_time(v):
    if v < LOW_CONGESTION_LIMIT:  return 20
    elif v < HIGH_CONGESTION_LIMIT: return 40
    else:        return 60

df['Congestion'] = df['Vehicles'].apply(congestion)

X = df[['Junction', 'hour', 'day']]
y = df['Vehicles']
y_class = df['Congestion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, yc_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))
rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))
lr_r2   = float(r2_score(y_test, lr_pred))
rf_r2   = float(r2_score(y_test, rf_pred))

rfc_acc = float(accuracy_score(yc_test, rfc.predict(X_test)))

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return send_from_directory(os.path.dirname(__file__), "index.html")

@app.route("/api/stats")
def stats():
    return jsonify({
        "total_records": int(len(df)),
        "junctions": 4,
        "date_range": {
            "start": str(df['DateTime'].min().date()),
            "end":   str(df['DateTime'].max().date())
        },
        "avg_vehicles": round(float(df['Vehicles'].mean()), 2),
        "max_vehicles": int(df['Vehicles'].max()),
        "congestion_dist": df['Congestion'].value_counts().to_dict(),
        "model_metrics": {
            "lr":  {"rmse": round(lr_rmse, 2), "r2": round(lr_r2, 4)},
            "rf":  {"rmse": round(rf_rmse, 2), "r2": round(rf_r2, 4)},
            "rfc_accuracy": round(rfc_acc, 4)
        }
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    junction = int(data.get("junction", 1)) - 1   # 0-indexed
    hour      = int(data.get("hour", 12))
    day       = int(data.get("day", 0))
    sample = [[junction, hour, day]]
    pred_vehicles  = float(rf.predict(sample)[0])
    pred_congestion = congestion(pred_vehicles)
    pred_signal     = signal_time(pred_vehicles)
    return jsonify({
        "predicted_vehicles": round(pred_vehicles, 1),
        "congestion_level":   pred_congestion,
        "signal_time_seconds": pred_signal
    })

@app.route("/api/heatmap")
def heatmap():
    junction = request.args.get("junction")
    heatmap_df = df
    if junction is not None:
        heatmap_df = df[df['Junction'] == int(junction) - 1]
    pivot = heatmap_df.pivot_table(values='Vehicles', index='hour', columns='day', aggfunc='mean')
    result = []
    for hour in range(24):
        for day in range(7):
            val = pivot.get(day, {}).get(hour, 0)
            vehicles = round(float(val) if val else 0, 1)
            result.append({
                "hour": hour,
                "day": int(day),
                "vehicles": vehicles,
                "congestion_level": congestion(vehicles)
            })
    return jsonify(result)

@app.route("/api/junction_trend")
def junction_trend():
    junction = int(request.args.get("junction", 1)) - 1
    sub = df[df['Junction'] == junction]
    hourly = sub.groupby('hour')['Vehicles'].mean().reset_index()
    return jsonify([
        {"hour": int(r['hour']), "avg_vehicles": round(float(r['Vehicles']), 1)}
        for _, r in hourly.iterrows()
    ])

@app.route("/api/daily_pattern")
def daily_pattern():
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    daily = df.groupby('day')['Vehicles'].mean().reset_index()
    return jsonify([
        {"day": days[int(r['day'])], "avg_vehicles": round(float(r['Vehicles']), 1)}
        for _, r in daily.iterrows()
    ])

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
