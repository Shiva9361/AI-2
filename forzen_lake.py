import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from typing import cast, Tuple, Callable
from collections import defaultdict

import imageio
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut

import time
import matplotlib.pyplot as plt

N = 8
Timeout = 60  # in seconds :)

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


@func_set_timeout(Timeout)
def iterative_deepening_a_star(env: Env[Discrete, Discrete], x: int) -> Tuple[float, list[int]]:
    stack: list[Tuple[int, int, list[int]]] = [(0, x, [])]

    u = 0
    u_dash = 0
    finished = False

    # manhattan distance heuristic
    h: Callable[[int], int] = lambda state: abs(
        state // N - (N-1)) + abs(state % N - (N-1))

    while not finished:
        u = u_dash
        u_dash = float('inf')
        stack = [(0, x, [])]
        while stack:
            depth, x, path = stack.pop()
            for action in range(cast(int, env.action_space.n)):
                transitions = env.unwrapped.P[x][action]

                for _, y, _, _ in transitions:
                    if depth + 1 + h(y) <= u and x != y:
                        if y == N**2-1:
                            moves = path + [action]
                            finished = True
                            break
                        else:
                            stack.append((depth + 1, y, path + [action]))
                    if depth + 1 + h(y) < u_dash:
                        u_dash = depth + 1 + h(y)
    return u, moves


def evaluate(env: Env[Discrete, Discrete], algorithm: Callable[[Env[Discrete, Discrete], int], Tuple[float, list[int]]]) -> None:
    moves: list[int] = []
    times: list[float] = []
    for _ in range(5):
        x, _ = env.reset()
        print("Starting from state:", x)
        start = time.time()
        try:
            u, moves = algorithm(env, cast(int, x))
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
    plt.title(f"{algorithm.__name__} on Frozen Lake {N}x{N}")
    plt.savefig(f'frozen_lake_{algorithm.__name__}_{N}x{N}.png')

    env.reset()
    frames: list = [env.render()]
    for action in moves:
        _ = env.step(cast(Discrete, action))
        frames.append(env.render())

    imageio.mimsave(
        f"frozen_lake_{algorithm.__name__}_{N}x{N}.gif", frames, duration=0.3)

    print(f"GIF saved as frozen_lake_{algorithm.__name__}_{N}x{N}.gif")


if __name__ == "__main__":
    evaluate(env, depth_first_branch_and_bound)
    evaluate(env, iterative_deepening_a_star)
