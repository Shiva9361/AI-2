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

N = 16
Timeout = 60*10  # in seconds :)
times_dfbnb: list[float] = []
times_ida_star: list[float] = []
env: Env[Discrete, Discrete] = gym.make('FrozenLake-v1',desc=generate_random_map(N), is_slippery=False, render_mode="rgb_array")

@func_set_timeout(Timeout)
def iterative_deepening_a_star(env: Env[Discrete, Discrete], x: int) -> Tuple[float, list[int]]:
    stack: list[Tuple[int, int, list[int]]] = [(0, x, [])]

    distance: defaultdict[int, float] = defaultdict(lambda: float('inf'))
    distance[x] = 0
    u = 0
    u_dash = 0
    finished = False

    # manhattan distance heuristic
    h: Callable[[int], int] = lambda state: abs(
        state // N - (N-1)) + abs(state % N - (N-1))
    
    u_dash = h(x)

    while not finished:
        u = u_dash
        u_dash = float('inf')
        stack = [(0, x, [])]
        while stack:
            depth, x, path = stack.pop()
            for action in range(cast(int, env.action_space.n)):
                transitions = env.unwrapped.P[x][action]

                for _, y, _, _ in transitions:
                    if depth + 1 + h(y) <= u and distance[y]>distance[x]+1 and x != y:
                        distance[y]=distance[x]+1
                        if y == N**2-1:
                            moves = path + [action]
                            finished = True
                            break
                        else:
                            stack.append((depth + 1, y, path + [action]))
                    if depth + 1 + h(y) < u_dash:
                        u_dash = depth + 1 + h(y)
    return u, moves

@func_set_timeout(Timeout)
def dfbnb(env: Env[Discrete, Discrete], x: int) -> Tuple[float, list[int]]:
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

def evaluate(env: Env[Discrete, Discrete], algorithm: Callable[[Env[Discrete, Discrete], int], Tuple[float, list[int]]]) -> None:
    """
    Evaluate the algorithm on the TSP environment.
    Args:
        env (Env[Discrete, Discrete]): The TSP environment.
        algorithm (Callable[[Env,Discrete,Discrete],Tuple[float,list[int]]]): The algorithm to evaluate.
    Returns:
        None
    """
    global times_ida_star, times_dfbnb  # <-- Add this line
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
    

    env.reset()
    frames: list = [env.render()]
    for action in moves:
        _ = env.step(cast(Discrete, action))
        frames.append(env.render())

    imageio.mimsave(
        f"results/frozen_lake/{algorithm.__name__}_on_frozen_lake_{N}x{N}.gif", frames, duration=0.3)

    print(f"GIF saved as {algorithm.__name__}_on_frozen_lake__{N}x{N}.gif")

    # Store in appropriate list based on algorithm name
    print(algorithm.__name__)
    if algorithm.__name__ == "iterative_deepening_a_star":
        times_ida_star = times.copy()
    else:
        times_dfbnb = times.copy()

    plt.plot(times, label=f'{algorithm.__name__}', marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Time taken (s)')
    # Add average line
    avg_time= sum(times) / len(times)
    plt.axhline(y=avg_time, color='blue', linestyle='--', label=f'Avg time: {avg_time:.4f}s')

    # Legend
    plt.legend()
    plt.title(f"Time of {algorithm.__name__} on Frozen Lake {N}x{N}")
    plt.savefig(f'results/frozen_lake/{algorithm.__name__}_on_frozen_lake_{N}x{N}.png')
    plt.clf()

def comparisonPlot():
    plt.plot(times_ida_star,color='blue', label=f'Ida* Time', marker='o')
    plt.plot(times_dfbnb,color='red',label=f'Dfbnb Time', marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Time taken (s)')
    # Add average line
    avg_time_ida_star = sum(times_ida_star) / len(times_ida_star)
    plt.axhline(y=avg_time_ida_star, color='blue', linestyle='--', label=f'Avg time ida*: {avg_time_ida_star:.4f}s')
    avg_time_dfbnb = sum(times_dfbnb) / len(times_dfbnb)
    plt.axhline(y=avg_time_dfbnb, color='orange', linestyle='--', label=f'Avg time dfbnb: {avg_time_dfbnb:.4f}s')

    # Legend
    plt.legend()
    plt.title(f"Time Comparison between on Frozen Lake {N}x{N}")
    plt.savefig(f'results/frozen_lake/dfbnb_vs_ida*_on_frozen_lake_{N}x{N}.png')

if __name__ == "__main__":
    evaluate(env, dfbnb)
    evaluate(env, iterative_deepening_a_star)
    comparisonPlot()
