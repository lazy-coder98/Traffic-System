import pytest
import numpy as np
import pandas as pd
from rl_agent import QLearningTrafficManager, congestion_label, signal_time
import config

def test_congestion_label():
    assert congestion_label(10) == "Low"
    assert congestion_label(config.LOW_CONGESTION_LIMIT) == "Medium"
    assert congestion_label(config.HIGH_CONGESTION_LIMIT) == "High"

def test_signal_time():
    assert signal_time(10) == 20
    assert signal_time(40) == 40
    assert signal_time(60) == 60

def test_qlearning_state_roundtrip():
    agent = QLearningTrafficManager(n_junctions=4)
    # Test all possible states
    for s in range(agent.n_states):
        junc, hb, dt, cong = agent._decode_state(s)
        encoded = agent._state_index(junc, hb, dt, cong)
        assert encoded == s, f"State mismatch: expected {s}, got {encoded}"

def test_reward_function():
    agent = QLearningTrafficManager()
    
    # Low traffic, action index 4 (70s signal) should have a huge penalty
    penalty_long_signal_empty_road = agent._reward(10, 4)
    assert penalty_long_signal_empty_road < -20
    
    # High traffic, action index 0 (15s signal) should have a huge penalty
    penalty_short_signal_busy_road = agent._reward(80, 0)
    assert penalty_short_signal_busy_road < -20
    
    # Medium traffic, right signal duration should get a positive bonus
    # 40 vehicles -> ideal around 42s. Action 2 (35s) or 3 (50s) should be good.
    reward_good = agent._reward(40, 2)
    assert reward_good > 0
