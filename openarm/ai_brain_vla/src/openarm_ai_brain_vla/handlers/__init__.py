"""One module per exposed member. Each reads the goal, applies the
contract's name and lane rules through the brain's core, looks through the
brain (`Brain.look`) or calls the manipulator, and returns the result
fields; the brain completes the goal and logs its end.

The `ZERO` dict in each action module is the fields the contract says are
zero or empty when success is false.
"""
