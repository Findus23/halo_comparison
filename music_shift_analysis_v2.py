# #Part two: the revenge of the shifts

import numpy as np

Lbox = 100.0

# #This is Adrian's solution, which we want to achieve as well
h_adrian = 0.6777
centre_adrian = np.array(
    [56.51984186, 49.40215384, 49.64235143]
)  # NOTE: as determined from his snapshot at z=0!
adrian_origin_in_parent_sim = (
    np.array([4.367988, 1.307087, 7.743252]) / h_adrian
)  # equal to his shift, so this should give the original parent sim coordinates:
parent_sim_coords_of_zoom_halo_adrian = centre_adrian - adrian_origin_in_parent_sim

print(
    f"zoom halo end coordinates in parent sim according to adrian:  {parent_sim_coords_of_zoom_halo_adrian}"
)

# #This is, step by step, Music's way of dealing with the shift:
# necessary for Music's calculation
shift_unit = 2
ncoarse = 1 << 7  # == 2 ** 7 (x << y == x * 2 ** y)
desired_box_centre = np.array([0.5, 0.5, 0.5])

# aka x0ref in music code:
left_corner_highres_boundary = np.array([0.530964, 0.476099, 0.579666])
# aka x1ref in music code:
right_corner_highres_boundary = np.array([0.601410, 0.558650, 0.645789])

# this seems to be the extent of the highres region in units such that it lies in [0...1[
lxref = right_corner_highres_boundary - left_corner_highres_boundary

# the centre of the highres region in Music:
xc = (left_corner_highres_boundary + 0.5 * lxref) % 1.0

# quick sanity check here
# Adrian's shift should make the centre of the zoom region the centre of the box, so by adding his inverse shift to desired_box_centre, we might get an idea of the IC coordinates of his zoom region in the parent sim
print(
    f"our calculated centre (at a=0.02): {xc * 100} | zoom halo IC coordinates in parent sim according to adrian: {desired_box_centre * 100 + adrian_origin_in_parent_sim}"
)

music_centre = np.array([58.29183329, 51.33582822, 51.69411051])
print(f"Music centre at a=1: {music_centre} | Adrian (at a=1): {centre_adrian}")

# back to Music
xshift = (desired_box_centre - xc) * ncoarse / shift_unit + desired_box_centre
xshift_without_int_conversion = xshift * shift_unit
xshift = xshift.astype(int) * shift_unit
print(
    f"Shift as stated by Music: {xshift} | Shift as desired by Music (without int conversion): {xshift_without_int_conversion}"
)

# convert xshifts
coarse_cell_size = Lbox / ncoarse
xshift_in_mpc = xshift * coarse_cell_size
xshift_without_int_conversion_in_mpc = xshift_without_int_conversion * coarse_cell_size
print(
    f"Shift as stated by Music [Mpc]: {xshift_in_mpc} | Shift as desired by Music (without int conversion) [Mpc]: {xshift_without_int_conversion_in_mpc}"
)

# apply shifts to centre
parent_sim_coords_of_zoom_halo_music = xc * 100 + xshift_in_mpc
parent_sim_coords_of_zoom_halo_music_without_int_conversion = (
    xc * 100 + xshift_without_int_conversion_in_mpc
)

print(
    f"zoom halo IC coordinates in zoom sim according to Music: {parent_sim_coords_of_zoom_halo_music}"
)
print(
    f"zoom halo IC coordinates in zoom sim according to Music without int conversion: {parent_sim_coords_of_zoom_halo_music_without_int_conversion}"
)
print(
    f"zoom halo end coordinates in parent sim according to Music: {music_centre + xshift_in_mpc}"
)
print(
    f"zoom halo end coordinates in parent sim according to Music without int conversion: {music_centre + xshift_without_int_conversion_in_mpc}"
)

print("--------------")

print(xshift_in_mpc + adrian_origin_in_parent_sim)
print(xshift_without_int_conversion_in_mpc + adrian_origin_in_parent_sim)
print(music_centre - centre_adrian)
print(music_centre + xshift_in_mpc - parent_sim_coords_of_zoom_halo_adrian)
print(
    music_centre
    + xshift_without_int_conversion_in_mpc
    - parent_sim_coords_of_zoom_halo_adrian
)
