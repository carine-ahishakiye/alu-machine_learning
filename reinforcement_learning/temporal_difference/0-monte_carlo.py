#!/usr/bin/env python3
"""Monte Carlo algorithm for value estimation"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm to estimate a value function.

    Args:
        env: openAI environment instance
        V (numpy.ndarray): shape (s,) containing the value estimate
        policy (function): takes in a state and returns the next action
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): learning rate
        gamma (float): discount rate

    Returns:
        V (numpy.ndarray): the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if done:
                break

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V