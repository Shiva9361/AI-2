import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from typing import cast, Tuple
from collections import defaultdict

import imageio
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut

import time
import matplotlib.pyplot as plt

N = 8
Timeout = 10  # in seconds :)

env: Env[Discrete, Discrete] = gym.make('FrozenLake-v1',
                                        desc=generate_random_map(N), is_slippery=False, render_mode="rgb_array")
# desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array")


@func_set_timeout(Timeout)
def depth_first_branch_and_bound(env: Env[Discrete, Discrete], x: int) -> Tuple[float, list[int]]:
    stack: list[Tuple[int, list[int]]] = [(x, [])]

    distance: defaultdict[int, float] = defaultdict(lambda: float('inf'))
    distance[x] = 0

    u = float('inf')

    moves: list[int] = []

    while stack:
        x, path = stack.pop()
        for action in range(cast(int, env.action_space.n)):
            transitions = env.unwrapped.P[x][action]

            for _, y, _, _ in transitions:
                if distance[x] + 1 < u and distance[y] > distance[x] + 1 and x != y:
                    distance[y] = distance[x] + 1

                    if y == N**2-1:
                        if (u > distance[y]):
                            u = distance[y]
                            moves = path + [action]
                    else:
                        stack.append((y, path + [action]))
    return u, moves


moves: list[int] = []
times: list[float] = []
for _ in range(5):
    x, _ = env.reset()
    print("Starting from state:", x)
    start = time.time()
    try:
        u, moves = depth_first_branch_and_bound(env, cast(int, x))
    except FunctionTimedOut:
        print("Timed out...")
        continue
    stop = time.time()
    times.append(stop - start)
    print("Time taken:", times[-1])
    print("Distance to goal:", u)

plt.plot(times)
plt.xlabel('Run')
plt.ylabel('Time taken (s)')
plt.title(f"Depth First Branch and Bound on Frozen Lake {N}x{N}")
plt.savefig(f'frozen_lake_dfbnb_{N}x{N}.png')

env.reset()
frames: list = [env.render()]
for action in moves:
    _ = env.step(cast(Discrete, action))
    frames.append(env.render())

imageio.mimsave(f"frozen_lake_dfbnb_{N}x{N}.gif", frames, duration=0.3)

print("GIF saved as frozen_lake_dfbnb.gif")
