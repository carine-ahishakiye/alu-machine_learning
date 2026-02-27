#!/usr/bin/env python3
"""TD(lambda) algorithm for value estimation"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm to estimate a value function.

    Args:
        env: openAI environment instance
        V (numpy.ndarray): shape (s,) containing the value estimate
        policy (function): takes in a state and returns the next action
        lambtha (float): eligibility trace factor
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): learning rate
        gamma (float): discount rate

    Returns:
        V (numpy.ndarray): the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        Et = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            delta = reward + gamma * V[next_state] * (1 - done) - V[state]
            Et[state] += 1

            V += alpha * delta * Et
            Et *= gamma * lambtha

            state = next_state
            if done:
                break

    return V