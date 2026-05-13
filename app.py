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
import json
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

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

# ── Q-Learning Traffic Manager ──────────────────────────────────────
class QLearningTrafficManager:
    """
    Offline Q-Learning agent that learns optimal signal timing policies
    from historical traffic data.

    State:  (junction, hour_bucket, day_type, congestion_level)
    Action: signal duration index → [15, 25, 35, 50, 70] seconds
    Reward: penalises mismatch between signal allocation and demand
    """

    ACTIONS = [15, 25, 35, 50, 70]             # signal durations (seconds)
    HOUR_BUCKETS = [(0,6), (6,12), (12,18), (18,24)]  # night / morning / afternoon / evening
    HOUR_BUCKET_NAMES = ['Night', 'Morning', 'Afternoon', 'Evening']
    DAY_TYPES = ['Weekday', 'Weekend']
    CONGESTION_LEVELS = ['Low', 'Medium', 'High']
    ACTION_LABELS = [
        'Reduce green phase',
        'Normal operation',
        'Slight extension',
        'Extend green phase',
        'Maximum green + reroute advisory'
    ]
    MANAGEMENT_ACTIONS = [
        'Reduce',
        'Normal',
        'Extend',
        'Extend+',
        'Reroute'
    ]

    def __init__(self, n_junctions=4, alpha=0.1, gamma=0.95, epsilon=0.15, episodes=800):
        self.n_junctions = n_junctions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.n_actions = len(self.ACTIONS)

        # State dimensions
        self.n_hour_buckets = len(self.HOUR_BUCKETS)
        self.n_day_types = len(self.DAY_TYPES)
        self.n_congestion = len(self.CONGESTION_LEVELS)

        # Q-table: (junction × hour_bucket × day_type × congestion) × actions
        self.n_states = n_junctions * self.n_hour_buckets * self.n_day_types * self.n_congestion
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Training history
        self.reward_history = []
        self.converged_at = 0
        self.total_reward = 0.0

    def _hour_bucket(self, hour):
        for i, (lo, hi) in enumerate(self.HOUR_BUCKETS):
            if lo <= hour < hi:
                return i
        return 0

    def _day_type(self, day):
        return 0 if day < 5 else 1  # 0=weekday, 1=weekend

    def _congestion_idx(self, vehicles):
        if vehicles < LOW_CONGESTION_LIMIT:
            return 0
        elif vehicles < HIGH_CONGESTION_LIMIT:
            return 1
        return 2

    def _state_index(self, junction, hour_bucket, day_type, congestion_idx):
        return (
            junction * (self.n_hour_buckets * self.n_day_types * self.n_congestion) +
            hour_bucket * (self.n_day_types * self.n_congestion) +
            day_type * self.n_congestion +
            congestion_idx
        )

    def _decode_state(self, state_idx):
        cong = state_idx % self.n_congestion
        rem = state_idx // self.n_congestion
        dt = rem % self.n_day_types
        rem = rem // self.n_day_types
        hb = rem % self.n_hour_buckets
        junc = rem // self.n_hour_buckets
        return junc, hb, dt, cong

    def _reward(self, vehicles, action_idx):
        """
        Reward function: penalises over-allocation (wasted green on empty roads)
        and under-allocation (short green on congested roads).
        """
        signal_dur = self.ACTIONS[action_idx]
        # Ideal signal based on demand (continuous)
        ideal = 10 + (vehicles / 80.0) * 65  # maps 0→10s, 80→75s
        ideal = np.clip(ideal, 10, 75)

        # Penalty: quadratic mismatch
        mismatch = abs(signal_dur - ideal)
        reward = -0.5 * (mismatch ** 1.3)

        # Bonus for being within ±8s of ideal
        if mismatch <= 8:
            reward += 15.0

        # Extra penalty for very short signals on busy roads
        if vehicles > HIGH_CONGESTION_LIMIT and signal_dur < 35:
            reward -= 20.0

        # Extra penalty for very long signals on empty roads
        if vehicles < LOW_CONGESTION_LIMIT and signal_dur > 50:
            reward -= 15.0

        return reward

    def train(self, dataframe):
        """Train the Q-table offline using experience from the dataset."""
        rng = np.random.default_rng(42)
        records = dataframe[['Junction', 'hour', 'day', 'Vehicles']].values
        n_records = len(records)

        for episode in range(self.episodes):
            episode_reward = 0.0
            # Decay epsilon over training
            eps = self.epsilon * max(0.1, 1.0 - episode / self.episodes)

            # Sample a batch of experiences from the dataset
            indices = rng.choice(n_records, size=min(200, n_records), replace=False)

            for idx in indices:
                junction, hour, day, vehicles = records[idx]
                junction = int(junction)
                hour_bucket = self._hour_bucket(int(hour))
                day_type = self._day_type(int(day))
                cong_idx = self._congestion_idx(vehicles)

                state = self._state_index(junction, hour_bucket, day_type, cong_idx)

                # Epsilon-greedy action selection
                if rng.random() < eps:
                    action = rng.integers(0, self.n_actions)
                else:
                    action = int(np.argmax(self.q_table[state]))

                reward = self._reward(vehicles, action)
                episode_reward += reward

                # Next state: sample a nearby record for temporal continuity
                next_idx = min(idx + 1, n_records - 1)
                nj, nh, nd, nv = records[next_idx]
                next_state = self._state_index(
                    int(nj),
                    self._hour_bucket(int(nh)),
                    self._day_type(int(nd)),
                    self._congestion_idx(nv)
                )

                # Q-update
                best_next = np.max(self.q_table[next_state])
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * best_next - self.q_table[state, action]
                )

            self.reward_history.append(episode_reward)

        # Detect convergence (reward stabilisation)
        if len(self.reward_history) > 50:
            recent = self.reward_history[-50:]
            for i in range(len(self.reward_history) - 50):
                window = self.reward_history[i:i+50]
                if np.std(window) < 0.05 * abs(np.mean(window) + 1e-9):
                    self.converged_at = i + 50
                    break
            else:
                self.converged_at = self.episodes
        self.total_reward = float(np.sum(self.reward_history))

    def get_recommendation(self, junction, hour, day, vehicles=None):
        """
        Get the RL-recommended management action for a given scenario.
        If vehicles is None, uses the RF model to predict it.
        """
        hour_bucket = self._hour_bucket(hour)
        day_type = self._day_type(day)

        if vehicles is None:
            vehicles = float(rf.predict([[junction, hour, day]])[0])

        cong_idx = self._congestion_idx(vehicles)
        state = self._state_index(junction, hour_bucket, day_type, cong_idx)

        # Best action from Q-table
        q_values = self.q_table[state].copy()
        best_action = int(np.argmax(q_values))
        signal_dur = self.ACTIONS[best_action]

        # Rule-based comparison
        rule_signal = signal_time(vehicles)

        # Phase allocation (green / amber / red ratios based on signal duration)
        cycle_time = max(signal_dur * 2.2, 60)  # full cycle
        green_ratio = round(signal_dur / cycle_time * 100, 1)
        amber_time = 4  # fixed
        red_time = round(cycle_time - signal_dur - amber_time, 1)

        # Priority score: 0-100 based on congestion and Q-value magnitude
        q_range = float(np.max(q_values) - np.min(q_values)) if np.max(q_values) != np.min(q_values) else 1.0
        urgency = (cong_idx / 2.0) * 60 + (1.0 - (q_values[best_action] - np.min(q_values)) / q_range) * 40
        priority = round(min(100, max(0, urgency)), 1)

        return {
            'optimal_signal_seconds': signal_dur,
            'rule_based_signal_seconds': rule_signal,
            'improvement_seconds': rule_signal - signal_dur,
            'action_index': best_action,
            'action_label': self.ACTION_LABELS[best_action],
            'management_action': self.MANAGEMENT_ACTIONS[best_action],
            'priority_score': priority,
            'phase_allocation': {
                'green_pct': green_ratio,
                'amber_seconds': amber_time,
                'red_seconds': max(0, red_time),
                'cycle_seconds': round(cycle_time, 1)
            },
            'congestion_level': self.CONGESTION_LEVELS[cong_idx],
            'hour_bucket': self.HOUR_BUCKET_NAMES[hour_bucket],
            'day_type': self.DAY_TYPES[day_type],
            'q_values': [round(float(v), 2) for v in q_values]
        }

    def get_policy_summary(self):
        """Return summary of the learned policy for all states."""
        summary = []
        for s in range(self.n_states):
            junc, hb, dt, cong = self._decode_state(s)
            best_action = int(np.argmax(self.q_table[s]))
            summary.append({
                'junction': junc + 1,
                'hour_bucket': self.HOUR_BUCKET_NAMES[hb],
                'day_type': self.DAY_TYPES[dt],
                'congestion': self.CONGESTION_LEVELS[cong],
                'optimal_signal': self.ACTIONS[best_action],
                'management_action': self.MANAGEMENT_ACTIONS[best_action],
                'q_value': round(float(np.max(self.q_table[s])), 2)
            })
        return summary


# Train the RL agent
rl_manager = QLearningTrafficManager(n_junctions=4, alpha=0.1, gamma=0.95, epsilon=0.15, episodes=800)
rl_manager.train(df)

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
    # RL management recommendation
    rl_rec = rl_manager.get_recommendation(junction, hour, day, pred_vehicles)
    return jsonify({
        "predicted_vehicles": round(pred_vehicles, 1),
        "congestion_level":   pred_congestion,
        "signal_time_seconds": pred_signal,
        "rl_recommendation": rl_rec
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

# ── Management API routes ─────────────────────────────────────────
@app.route("/api/management/status")
def management_status():
    """Current RL policy summary for all junctions at the current time."""
    from datetime import datetime
    now = datetime.now()
    current_hour = now.hour
    current_day = now.weekday()

    junction_status = []
    for j in range(4):
        pred_v = float(rf.predict([[j, current_hour, current_day]])[0])
        rec = rl_manager.get_recommendation(j, current_hour, current_day, pred_v)
        junction_status.append({
            'junction': j + 1,
            'predicted_vehicles': round(pred_v, 1),
            **rec
        })

    # Sort by priority (highest first)
    junction_status.sort(key=lambda x: x['priority_score'], reverse=True)

    return jsonify({
        'timestamp_hour': current_hour,
        'timestamp_day': current_day,
        'junctions': junction_status,
        'training_metrics': {
            'episodes': rl_manager.episodes,
            'converged_at': rl_manager.converged_at,
            'total_reward': round(rl_manager.total_reward, 1),
            'avg_reward_last_50': round(float(np.mean(rl_manager.reward_history[-50:])), 1) if rl_manager.reward_history else 0,
            'reward_history_sample': [round(float(r), 1) for r in rl_manager.reward_history[::max(1, len(rl_manager.reward_history)//40)]]
        }
    })


@app.route("/api/management/recommend", methods=["POST"])
def management_recommend():
    """Get RL recommendation for a specific junction/hour/day."""
    data = request.json
    junction = int(data.get("junction", 1)) - 1
    hour = int(data.get("hour", 12))
    day = int(data.get("day", 0))

    pred_v = float(rf.predict([[junction, hour, day]])[0])
    rec = rl_manager.get_recommendation(junction, hour, day, pred_v)

    return jsonify({
        'junction': junction + 1,
        'hour': hour,
        'day': day,
        'predicted_vehicles': round(pred_v, 1),
        **rec
    })


@app.route("/api/management/policy_heatmap")
def management_policy_heatmap():
    """Return the RL-recommended signal durations as a heatmap (hour × day) per junction."""
    junction = int(request.args.get("junction", 1)) - 1
    result = []
    for hour in range(24):
        for day in range(7):
            pred_v = float(rf.predict([[junction, hour, day]])[0])
            rec = rl_manager.get_recommendation(junction, hour, day, pred_v)
            rule_sig = signal_time(pred_v)
            result.append({
                'hour': hour,
                'day': day,
                'rl_signal': rec['optimal_signal_seconds'],
                'rule_signal': rule_sig,
                'management_action': rec['management_action'],
                'priority_score': rec['priority_score'],
                'predicted_vehicles': round(pred_v, 1)
            })
    return jsonify(result)


@app.route("/api/management/comparison")
def management_comparison():
    """Compare RL policy vs rule-based across all junctions and time periods."""
    comparison = []
    total_rl = 0
    total_rule = 0
    total_scenarios = 0

    for j in range(4):
        junction_data = {'junction': j + 1, 'periods': []}
        for hb_idx, (hb_name) in enumerate(rl_manager.HOUR_BUCKET_NAMES):
            hb_lo, hb_hi = rl_manager.HOUR_BUCKETS[hb_idx]
            # Average over hours in bucket and all days
            rl_signals = []
            rule_signals = []
            for hour in range(hb_lo, hb_hi):
                for day in range(7):
                    pred_v = float(rf.predict([[j, hour, day]])[0])
                    rec = rl_manager.get_recommendation(j, hour, day, pred_v)
                    rl_signals.append(rec['optimal_signal_seconds'])
                    rule_signals.append(signal_time(pred_v))

            avg_rl = round(float(np.mean(rl_signals)), 1)
            avg_rule = round(float(np.mean(rule_signals)), 1)
            total_rl += sum(rl_signals)
            total_rule += sum(rule_signals)
            total_scenarios += len(rl_signals)

            junction_data['periods'].append({
                'period': hb_name,
                'avg_rl_signal': avg_rl,
                'avg_rule_signal': avg_rule,
                'difference': round(avg_rule - avg_rl, 1)
            })
        comparison.append(junction_data)

    efficiency = round((1 - total_rl / max(total_rule, 1)) * 100, 1) if total_rule > 0 else 0

    return jsonify({
        'comparison': comparison,
        'overall': {
            'avg_rl_signal': round(total_rl / max(total_scenarios, 1), 1),
            'avg_rule_signal': round(total_rule / max(total_scenarios, 1), 1),
            'signal_efficiency_pct': efficiency,
            'total_scenarios': total_scenarios
        }
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
