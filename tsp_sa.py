import time
from func_timeout.exceptions import FunctionTimedOut
from func_timeout import func_set_timeout
from numpy.typing import NDArray
from numpy import dtype, unsignedinteger
from typing import Tuple,Callable
from TSP.gym_vrp.envs import TSPEnv

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import math
import imageio
import matplotlib
matplotlib.use('Agg')

TIMEOUT = 60*10
NODES = 20

UnsignedIntegerArray = NDArray[np.uint8 | np.uint16 | np.uint32 | np.uint64]

def visualize_route(env: TSPEnv, route: list[int], figsize: Tuple[int, int] = (10, 10), dpi: int = 100) -> UnsignedIntegerArray:
    """
    Visualize the TSP route on a 2D plot.
    Args:
        env (TSPEnv): The TSP environment.
        route (list[int]): The route to visualize.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 10).
        dpi (int, optional): Dots per inch. Defaults to 100.
    Returns:
        np.ndarray: The image array of the plot."""

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    positions = env.sampler.get_graph_positions()[0]
    depot_idx = env.depots[0][0]

    ax.scatter(positions[:, 0], positions[:, 1],
               c='#1f77b4', s=50, label='Cities')
    ax.scatter(positions[depot_idx, 0], positions[depot_idx, 1],
               c='red', s=150, marker='*', label='Depot')

    for i, (x, y) in enumerate(positions):
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8)

    route_positions = positions[route]
    ax.plot(route_positions[:, 0], route_positions[:, 1],
            '--', color='gray', linewidth=0.5, alpha=0.8)

    for i in range(len(route)-1):
        start = route_positions[i]
        end = route_positions[i+1]
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color='#2ca02c',
                                    lw=1.5, alpha=0.8))

    ax.set_title(f"TSP Route Visualization ({len(route)} cities)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)

    img_data, _ = canvas.print_to_buffer()
    img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(
        height, width, 4)[..., :3]
    plt.close(fig)

    return img_array


def generate_neighbor(solution: list[int]) -> list[int]:
    """
    Generate a neighbor solution by reversing a segment of the route
    Args:
        solution (list[int]): The current solution.
    Returns:
        list[int]: A neighbor solution.
    """
    neighbor = solution.copy()
    if len(neighbor) > 3:
        i, j = sorted(np.random.choice(len(neighbor)-1, 2, replace=False) + 1)
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
    return neighbor

def calculate_route_distance(env: TSPEnv, route: list[int]) -> float:
    """
    Calculate the total distance of the route.
    Args:
        env (TSPEnv): The TSP environment.
        route (list[int]): The route to evaluate.
    Returns:
        float: The total distance of the route.
    """
    total = 0.0
    for i in range(len(route)-1):
        edges = np.array([[route[i], route[i+1]]])
        total += env.sampler.get_distances(edges)[0]
    return total

@func_set_timeout(TIMEOUT)
def simulated_annealing(env: TSPEnv, iter_no:int = 1, max_iter: int = 1000, initial_temp: float=100.0, cooling_rate:float=0.995, save: bool = False) -> Tuple[list[int], float]:
    """
    Simulated Annealing algorithm for solving the TSP.
    Args:
        env (TSPEnv): The TSP environment.
        iter_no (int,optional): Which Iteration it is , Defaults to 1
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        initial_temp (float, optional): Initial temperature. Defaults to 100.0.
        cooling_rate (float, optional): Rate at which temperature decreases. Defaults to 0.995.
        save (bool, optional): Whether to save the animation. Defaults to False.

    Returns:
        Tuple[list[int], float]: Best route and its distance.
    """
    depot = int(env.depots[0][0])

    # Initialize current solution
    nodes = list(range(env.num_nodes))
    nodes.remove(depot)
    current_route = [depot] + np.random.permutation(nodes).tolist()
    current_dist = calculate_route_distance(env, current_route)

    best_route, best_dist = current_route, current_dist
    temperature = initial_temp
    frames = []

    for _ in range(max_iter):

        neighbor = generate_neighbor(current_route)
        neighbor_dist = calculate_route_distance(env, neighbor)

        delta = neighbor_dist - current_dist


        # If better solution OR with some probability
        if delta < 0 or np.random.rand() < math.exp(-delta / temperature):
            current_route, current_dist = neighbor, neighbor_dist

            if current_dist < best_dist:
                best_route, best_dist = current_route, current_dist
                if save:
                    frames.append(visualize_route(env, best_route))

        temperature *= cooling_rate
        if temperature < 1e-5:
            break

    if save:
        imageio.mimsave(f'tsp_simulated_annealing_{iter_no}.gif', frames, fps=3)

    return best_route, best_dist


def evaluate_algorithm(env: TSPEnv, algorithm: Callable[[TSPEnv, int, int], Tuple[list[int], float]], save:bool=False) -> None:
    """
    Evaluate the algorithm on the TSP environment.
    Args:
        env (TSPEnv): The TSP environment.
        algorithm (Callable[[TSPEnv,int,int],Tuple[list[int],float]]): The algorithm to evaluate.
    Returns:
        None
    """
    env.reset()
    times = []
    for i in range(5):

        try:
            start = time.time()
            optimal_route, distance = algorithm(env,iter_no=i,save=save)
            end = time.time()
            times.append(end - start)
            print(f"Execution time: {end - start:.2f} seconds")
            print(f"Optimal route: {optimal_route}\nDistance: {distance:.2f}")
        except FunctionTimedOut:
            print(f"{algorithm.__name__} timed out!")
        finally:
            plt.plot(range(1, len(times)+1), times, marker='o')
            plt.xlabel("Execution Index")
            plt.ylabel("Execution Time (seconds)")
            plt.title(f"{algorithm.__name__} Execution Time")
            plt.grid()
            plt.savefig(f"{algorithm.__name__}_execution_time.png")
            plt.close()


if __name__ == "__main__":
    env = TSPEnv(num_nodes=NODES)
    evaluate_algorithm(env, simulated_annealing,save=True)
    env.close()
    print("Execution completed.")
