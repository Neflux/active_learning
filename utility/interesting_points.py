import json
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random
from poisson_disc import Grid

spawn_area = 20
threshold = 1
step = 5

spawn_coords = np.stack(np.meshgrid(
    np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1),
    np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1)
)).reshape([2, -1]).T 

n_targets = len(spawn_coords) // 2

grid = Grid(threshold, spawn_area, spawn_area)
points = grid.poisson(None, static=spawn_coords)
targets = np.array(points[len(spawn_coords):])-spawn_area/2
np.random.shuffle(targets)
targets = targets[:n_targets]

spawn_coords -= spawn_area / 2

plt.scatter(*spawn_coords.T,color="r", marker="*")
plt.scatter(*targets.T,color="b", marker="1")
plt.title(f"Spawn points: {len(spawn_coords)}, pick up targets: {n_targets}, radius: {threshold}")
plt.show()