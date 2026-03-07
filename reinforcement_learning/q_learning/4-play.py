#!/usr/bin/env python3
"""Module for playing an episode with the trained agent."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        the total rewards for the episode
    """
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    env.render()

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        result = env.step(action)
        new_state, reward, done = result[0], result[1], result[2]
        env.render()
        total_reward += reward
        state = new_state

        if done:
            break

    return total_reward