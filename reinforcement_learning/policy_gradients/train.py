#!/usr/bin/env python3
"""
Training function using Monte-Carlo Policy Gradient (REINFORCE)
"""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )

    scores = []

    for episode in range(nb_episodes):
        state = env.reset()[None, :]

        episode_rewards = []
        episode_gradients = []

        done = False

        while not done:
            if show_result and (episode % 1000 == 0):
                env.render()

            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]

            episode_rewards.append(reward)
            episode_gradients.append(gradient)

            state = next_state

        score = sum(episode_rewards)
        scores.append(score)

        T = len(episode_rewards)
        G = 0
        returns = np.zeros(T)
        for t in reversed(range(T)):
            G = episode_rewards[t] + gamma * G
            returns[t] = G

        for t in range(T):
            weight += alpha * episode_gradients[t] * returns[t]

        print(f"Episode: {episode + 1} Score: {score}", end="\r", flush=False)

    return scores
