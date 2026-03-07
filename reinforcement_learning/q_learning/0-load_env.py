#!/usr/bin/env python3
"""Module to load the FrozenLake environment."""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from OpenAI's gym.

    Args:
        desc: None or a list of lists containing a custom map description
        map_name: None or a string containing the pre-made map to load
        is_slippery: boolean to determine if the ice is slippery

    Returns:
        the environment
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env