import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from typing import cast
from collections import defaultdict
import time

env: Env[Discrete, Discrete] = gym.make('FrozenLake-v1',  # type: ignore
                                        desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array")


def depth_first_branch_and_bound(env: Env[Discrete, Discrete], x: int) -> float:
    stack: list[int] = [x]

    distance: defaultdict[int, float] = defaultdict(lambda: float('inf'))
    distance[x] = 0

    u = float('inf')

    while stack:
        x = stack.pop()
        for action in range(cast(int, env.action_space.n)):  # type: ignore
            transitions = env.unwrapped.P[x][action]  # type: ignore

            for _, y, _, _ in transitions:  # type: ignore
                if distance[x] + 1 < u and distance[y] > distance[x] + 1 and x != y:
                    distance[y] = distance[x] + 1
                    if y == 63:
                        u = min(u, distance[y])
                    else:
                        stack.append(y)  # type: ignore
    return u


x, _ = env.reset()
print("Starting from state:", x)
start = time.time()
u = depth_first_branch_and_bound(env, cast(int, x))
print("Time taken:", time.time() - start)
print("Distance to goal:", u)
