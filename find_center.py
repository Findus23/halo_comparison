import hashlib
import json

import numpy as np
import pandas as pd

from utils import print_progress

cache_file = "center_cache.json"

try:
    with open(cache_file, "r") as f:
        center_cache = json.load(f)
except FileNotFoundError:
    center_cache = {}


def find_center(df: pd.DataFrame, center: np.ndarray, initial_radius=1):
    # plt.figure()
    all_particles = df[["X", "Y", "Z"]].to_numpy()
    hash = hashlib.sha256(np.ascontiguousarray(all_particles).data).hexdigest()
    if hash in center_cache:
        return np.array(center_cache[hash])
    radius = initial_radius
    center_history = []
    i = 0
    while True:
        center_history.append(center)
        distances = np.linalg.norm(all_particles - center, axis=1)
        in_radius_particles = all_particles[distances < radius]
        num_particles = in_radius_particles.shape[0]
        print_progress(i, "?", f"n={num_particles}, r={radius}, c={center}")
        if num_particles < 10:
            break
        center_of_mass = in_radius_particles.mean(axis=0)
        new_center = (center_of_mass + center) / 2
        shift = np.linalg.norm(center - new_center)
        radius = max(2 * shift, radius * 0.9)
        center = new_center
        i += 1
    center_history = np.array(center_history)
    # print(center_history)
    # plt.scatter(center_history[::, 0], center_history[::, 1], c=range(len(center_history[::, 1])))
    # plt.colorbar(label="step")
    # plt.show()
    print()
    center_cache[hash] = center.tolist()
    with open(cache_file, "w") as f:
        json.dump(center_cache, f)
    return center
