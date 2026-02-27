# Temporal Difference Learning

## Description

This directory contains implementations of Temporal Difference (TD) reinforcement learning algorithms, including Monte Carlo, TD(0), and TD(λ) methods applied to OpenAI Gym environments.

## Requirements

- Python 3.5+
- NumPy
- OpenAI Gym

## Files

| File | Description |
|------|-------------|
| `0-monte_carlo.py` | Monte Carlo algorithm for value function estimation |

## Tasks

### 0. Monte Carlo

Performs the Monte Carlo algorithm to estimate a value function.

```python
def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)
```

**Parameters:**
- `env`: OpenAI environment instance
- `V`: `numpy.ndarray` of shape `(s,)` containing the value estimate
- `policy`: function that takes a state and returns the next action
- `episodes`: total number of episodes to train over
- `max_steps`: maximum number of steps per episode
- `alpha`: learning rate
- `gamma`: discount rate

**Returns:** `V`, the updated value estimate

**Usage:**
```bash
$ ./0-main.py
```