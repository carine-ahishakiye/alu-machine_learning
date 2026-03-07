#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent."""
import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.core import Processor


class AtariProcessor(Processor):
    """Processor for Atari environments."""

    def process_observation(self, observation):
        """Process observation to uint8 to save memory."""
        if isinstance(observation, tuple):
            observation = observation[0]
        return observation.astype('uint8')

    def process_state_batch(self, batch):
        """Normalize pixel values to [0, 1]."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip reward to [-1, 1]."""
        return np.clip(reward, -1., 1.)


# Environment setup
ENV_NAME = 'ALE/Breakout-v5'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


def build_model(input_shape, n_actions):
    """Build the CNN model for the DQN agent."""
    model = Sequential([
        Permute((2, 3, 1), input_shape=input_shape),
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(n_actions, activation='linear')
    ])
    return model


def main():
    """Main play function."""
    # Create environment with render
    env = gym.make(ENV_NAME, render_mode='human')

    n_actions = env.action_space.n
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    # Build model
    model = build_model(input_shape, n_actions)

    # Configure memory
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    # Use GreedyQPolicy for playing (always exploit)
    policy = GreedyQPolicy()

    # Configure processor
    processor = AtariProcessor()

    # Build DQN agent
    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Load trained weights
    dqn.load_weights('policy.h5')
    print("Loaded policy from policy.h5")

    # Play 5 episodes
    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()


if __name__ == '__main__':
    main()