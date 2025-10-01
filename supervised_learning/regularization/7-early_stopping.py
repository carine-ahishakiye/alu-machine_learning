#!/usr/bin/env python3
"""Early stopping function"""

def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if training should be stopped early.

    cost: current validation cost
    opt_cost: lowest recorded validation cost
    threshold: threshold for improvement
    patience: patience count for stopping
    count: current count of epochs without sufficient improvement

    Returns: (stop: bool, updated_count: int)
    """
    # Check if improvement is less than threshold
    if opt_cost - cost > threshold:
        # Significant improvement, reset count
        count = 0
        opt_cost = cost
    else:
        # Not enough improvement, increment count
        count += 1

    # Determine if we should stop
    stop = count >= patience

    return stop, count
