from pathlib import Path
from sys import argv

import numpy as np
from h5py import File

fraction = 0.01
num_steps = 60

file = Path(argv[1])


def main():
    f = File(file)
    f_out = File(file.with_suffix(".sampled.hdf5"), "w")

    outpart = f_out.create_group("PartType1")

    num_particles = f["Header"].attrs["NumPart_Total"][1]
    print(num_particles)
    chosen_particles = int(num_particles * fraction)
    parttype1 = f["PartType1"]
    steps = np.linspace(0, num_particles, num_steps)
    column_data = {}
    columns = ["Coordinates", "Velocities", "ParticleIDs", "Masses"]
    original_data = {}
    for column in columns:
        original_data[column] = parttype1[column]
        column_data[column] = []
    for i in range(len(steps) - 1):
        start = int(steps[i])
        end = int(steps[i + 1])
        print(start, end)

        chosen_rows = np.random.choice(end - start, chosen_particles // num_steps)

        for column in columns:
            data = original_data[column][start:end]
            column_data[column].append(data[chosen_rows])

    for column in columns:
        if column in {"ParticleIDs", "Masses"}:
            all_data = np.hstack(column_data[column])
        else:
            all_data = np.vstack(column_data[column])

        out_column = outpart.create_dataset(
            column, data=all_data, compression="gzip" if column == "Masses" else None
        )
    print(len(out_column))
    f_out.close()


if __name__ == "__main__":
    main()
