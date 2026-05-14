"""
TrafficIQ — Q-Learning Traffic Management Agent

Offline Q-Learning agent that learns optimal signal timing policies
from historical traffic data.

State:  (junction, hour_bucket, day_type, congestion_level)
Action: signal duration index → [15, 25, 35, 50, 70] seconds
Reward: penalises mismatch between signal allocation and demand
"""

import numpy as np
import config


class QLearningTrafficManager:

    ACTIONS = [15, 25, 35, 50, 70]  # signal durations (seconds)
    HOUR_BUCKETS = [(0, 6), (6, 12), (12, 18), (18, 24)]
    HOUR_BUCKET_NAMES = ["Night", "Morning", "Afternoon", "Evening"]
    DAY_TYPES = ["Weekday", "Weekend"]
    CONGESTION_LEVELS = ["Low", "Medium", "High"]
    ACTION_LABELS = [
        "Reduce green phase",
        "Normal operation",
        "Slight extension",
        "Extend green phase",
        "Maximum green + reroute advisory",
    ]
    MANAGEMENT_ACTIONS = ["Reduce", "Normal", "Extend", "Extend+", "Reroute"]

    def __init__(
        self,
        n_junctions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        episodes: int = 800,
    ):
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
        self.n_states = (
            n_junctions * self.n_hour_buckets * self.n_day_types * self.n_congestion
        )
        self.q_table = np.zeros((self.n_states, self.n_actions))

        # Training history
        self.reward_history: list[float] = []
        self.converged_at: int = 0
        self.total_reward: float = 0.0

    # ── State helpers ────────────────────────────────────────

    def _hour_bucket(self, hour: int) -> int:
        for i, (lo, hi) in enumerate(self.HOUR_BUCKETS):
            if lo <= hour < hi:
                return i
        return 0

    def _day_type(self, day: int) -> int:
        return 0 if day < 5 else 1  # 0 = weekday, 1 = weekend

    def _congestion_idx(self, vehicles: float) -> int:
        if vehicles < config.LOW_CONGESTION_LIMIT:
            return 0
        elif vehicles < config.HIGH_CONGESTION_LIMIT:
            return 1
        return 2

    def _state_index(
        self, junction: int, hour_bucket: int, day_type: int, congestion_idx: int
    ) -> int:
        return (
            junction * (self.n_hour_buckets * self.n_day_types * self.n_congestion)
            + hour_bucket * (self.n_day_types * self.n_congestion)
            + day_type * self.n_congestion
            + congestion_idx
        )

    def _decode_state(self, state_idx: int):
        cong = state_idx % self.n_congestion
        rem = state_idx // self.n_congestion
        dt = rem % self.n_day_types
        rem = rem // self.n_day_types
        hb = rem % self.n_hour_buckets
        junc = rem // self.n_hour_buckets
        return junc, hb, dt, cong

    # ── Reward function ──────────────────────────────────────

    def _reward(self, vehicles: float, action_idx: int) -> float:
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
        if vehicles > config.HIGH_CONGESTION_LIMIT and signal_dur < 35:
            reward -= 20.0

        # Extra penalty for very long signals on empty roads
        if vehicles < config.LOW_CONGESTION_LIMIT and signal_dur > 50:
            reward -= 15.0

        return reward

    # ── Training ─────────────────────────────────────────────

    def train(self, dataframe):
        """Train the Q-table offline using experience from the dataset."""
        rng = np.random.default_rng(42)
        records = dataframe[["Junction", "hour", "day", "Vehicles"]].values
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
                    self._congestion_idx(nv),
                )

                # Q-update
                best_next = np.max(self.q_table[next_state])
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * best_next - self.q_table[state, action]
                )

            self.reward_history.append(episode_reward)

        # Detect convergence (reward stabilisation)
        if len(self.reward_history) > 50:
            for i in range(len(self.reward_history) - 50):
                window = self.reward_history[i : i + 50]
                if np.std(window) < 0.05 * abs(np.mean(window) + 1e-9):
                    self.converged_at = i + 50
                    break
            else:
                self.converged_at = self.episodes
        self.total_reward = float(np.sum(self.reward_history))

    # ── Inference ────────────────────────────────────────────

    def get_recommendation(
        self, junction: int, hour: int, day: int, vehicles: float
    ) -> dict:
        """
        Get the RL-recommended management action for a given scenario.
        """
        hour_bucket = self._hour_bucket(hour)
        day_type = self._day_type(day)
        cong_idx = self._congestion_idx(vehicles)
        state = self._state_index(junction, hour_bucket, day_type, cong_idx)

        # Best action from Q-table
        q_values = self.q_table[state].copy()
        best_action = int(np.argmax(q_values))
        signal_dur = self.ACTIONS[best_action]

        # Rule-based comparison
        rule_signal = signal_time(vehicles)

        # Phase allocation (green / amber / red ratios)
        cycle_time = max(signal_dur * 2.2, 60)
        green_ratio = round(signal_dur / cycle_time * 100, 1)
        amber_time = 4  # fixed
        red_time = round(cycle_time - signal_dur - amber_time, 1)

        # Priority score: 0-100
        q_range = (
            float(np.max(q_values) - np.min(q_values))
            if np.max(q_values) != np.min(q_values)
            else 1.0
        )
        urgency = (cong_idx / 2.0) * 60 + (
            1.0 - (q_values[best_action] - np.min(q_values)) / q_range
        ) * 40
        priority = round(min(100, max(0, urgency)), 1)

        return {
            "optimal_signal_seconds": signal_dur,
            "rule_based_signal_seconds": rule_signal,
            "improvement_seconds": rule_signal - signal_dur,
            "action_index": best_action,
            "action_label": self.ACTION_LABELS[best_action],
            "management_action": self.MANAGEMENT_ACTIONS[best_action],
            "priority_score": priority,
            "phase_allocation": {
                "green_pct": green_ratio,
                "amber_seconds": amber_time,
                "red_seconds": max(0, red_time),
                "cycle_seconds": round(cycle_time, 1),
            },
            "congestion_level": self.CONGESTION_LEVELS[cong_idx],
            "hour_bucket": self.HOUR_BUCKET_NAMES[hour_bucket],
            "day_type": self.DAY_TYPES[day_type],
            "q_values": [round(float(v), 2) for v in q_values],
        }

    def get_policy_summary(self) -> list[dict]:
        """Return summary of the learned policy for all states."""
        summary = []
        for s in range(self.n_states):
            junc, hb, dt, cong = self._decode_state(s)
            best_action = int(np.argmax(self.q_table[s]))
            summary.append(
                {
                    "junction": junc + 1,
                    "hour_bucket": self.HOUR_BUCKET_NAMES[hb],
                    "day_type": self.DAY_TYPES[dt],
                    "congestion": self.CONGESTION_LEVELS[cong],
                    "optimal_signal": self.ACTIONS[best_action],
                    "management_action": self.MANAGEMENT_ACTIONS[best_action],
                    "q_value": round(float(np.max(self.q_table[s])), 2),
                }
            )
        return summary


# ── Standalone helpers (used by both train.py and app.py) ────
def signal_time(v: float) -> int:
    """Rule-based signal timing."""
    if v < config.LOW_CONGESTION_LIMIT:
        return 20
    elif v < config.HIGH_CONGESTION_LIMIT:
        return 40
    return 60


def congestion_label(v: float) -> str:
    """Map vehicle count to congestion level string."""
    if v < config.LOW_CONGESTION_LIMIT:
        return "Low"
    elif v < config.HIGH_CONGESTION_LIMIT:
        return "Medium"
    return "High"
