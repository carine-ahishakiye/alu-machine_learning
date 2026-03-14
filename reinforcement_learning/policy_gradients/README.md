# Policy Gradients

## Description
This directory contains the implementation of policy gradient methods for reinforcement learning.

## Files

| File | Description |
|------|-------------|
| `policy_gradient.py` | Contains the policy and gradient functions |

## Functions

### `policy(matrix, weight)`
Computes the policy with a weight of a matrix using softmax.

**Arguments:**
- `matrix`: state matrix of shape (1, n_features)
- `weight`: weight matrix of shape (n_features, n_actions)

**Returns:**
- Softmax probabilities over actions of shape (1, n_actions)

**Example:**
```python
import numpy as np
from policy_gradient import policy

weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01],
    [1.14374817e-04, 3.02332573e-01],
    [1.46755891e-01, 9.23385948e-02],
    [1.86260211e-01, 3.45560727e-01]
]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214, 0.01636746, 0.01196594, -0.03095031]
]))

res = policy(state, weight)
print(res)  # [[0.50351642 0.49648358]]
```

## Requirements
- Python 3.x
- NumPy

## Author
* ALU Machine Learning
