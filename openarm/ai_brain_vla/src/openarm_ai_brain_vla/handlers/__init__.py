"""One module per exposed member. Each reads the goal, applies the
contract's name and lane rules through the brain's core, calls the
perceiver or the manipulator, and returns the result fields; the brain
completes the goal.

The `ZERO` dict in each action module is the fields the contract says are
zero or empty when success is false.
"""
