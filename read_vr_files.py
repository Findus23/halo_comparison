from pathlib import Path

import h5py
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from readfiles import read_file

REFERENCE_RESOLUTION = 128

test_group = 2

directory = Path(f"/home/ben/sims/data_swift/monofonic_tests/DB2_{REFERENCE_RESOLUTION}_100")

group_catalog = h5py.File(directory / 'vroutput.catalog_groups')
# df = pd.DataFrame([group_catalog['Group_Size'], group_catalog['Offset'], group_catalog['Offset_unbound'], group_catalog['Parent_halo_ID']], columns=['group_size', 'offset', 'offset_unbound', 'parent_halo_id'])
df = pd.DataFrame({'group_size':group_catalog['Group_Size'], 'offset':group_catalog['Offset'], 'offset_unbound':group_catalog['Offset_unbound'], 'parent_halo_id':group_catalog['Parent_halo_ID']})


particle_catalog = h5py.File(directory / 'vroutput.catalog_particles')
particle_ids = particle_catalog['Particle_IDs'][:]

particle_catalog_unbound = h5py.File(directory / 'vroutput.catalog_particles.unbound')
particle_ids_unbound = particle_catalog_unbound['Particle_IDs'][:]

lower_offset = df['offset'][test_group]
higher_offset = df['offset'][test_group + 1]
particle_ids_in_halo = set(particle_ids[lower_offset:higher_offset])

lower_offset_unbound = df['offset_unbound'][test_group]
higher_offset_unbound = df['offset_unbound'][test_group + 1]
particle_ids_unbound_in_halo = set(particle_ids_unbound[lower_offset_unbound:higher_offset_unbound])

size = df['group_size'][test_group]

assert size == len(particle_ids_in_halo) + len(particle_ids_unbound_in_halo)


    # particles_in_ref_halo = df_ref.loc[df_ref["FOFGroupIDs"] == index]

particles, _ = read_file(directory)
print(particles)
print(particle_ids_in_halo)
bound_particles = particles.loc(list(particle_ids_in_halo))
unbound_particles = particles.loc(particle_ids_unbound_in_halo)

plt.scatter(bound_particles['X'], bound_particles['Y'], label='bound')
plt.scatter(unbound_particles['X'], unbound_particles['Y'], label='unbound')
plt.legend()
plt.show()











print(group_sizes)

