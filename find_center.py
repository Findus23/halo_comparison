import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_center(df: pd.DataFrame, center: np.ndarray, initial_radius=1):
    # plt.figure()
    all_particles = df[["X", "Y", "Z"]].to_numpy()
    radius = initial_radius
    center_history = []
    while True:
        center_history.append(center)
        distances = np.linalg.norm(all_particles - center, axis=1)
        in_radius_particles = all_particles[distances < radius]
        num_particles = in_radius_particles.shape[0]
        print("num_particles", num_particles)
        if num_particles < 10:
            break
        center_of_mass = in_radius_particles.mean(axis=0)
        new_center = (center_of_mass + center) / 2
        print("new center", new_center)
        shift = np.linalg.norm(center - new_center)
        radius = max(2 * shift, radius * 0.9)
        print("radius", radius)
        center = new_center
    center_history = np.array(center_history)
    # print(center_history)
    # plt.scatter(center_history[::, 0], center_history[::, 1], c=range(len(center_history[::, 1])))
    # plt.colorbar(label="step")
    # plt.show()
    return center
