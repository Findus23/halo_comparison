import numpy as np

# import math

# aka x0ref in music code:
left_corner_highres_boundary = np.array([0.530964, 0.476099, 0.579666])
# aka x1ref in music code:
right_corner_highres_boundary = np.array([0.601410, 0.558650, 0.645789])

# this seems to be the extent of the highres region in units such that it lies in [0...1[
lxref = right_corner_highres_boundary - left_corner_highres_boundary

# presumably the centre of the highres region:
# xc = left_corner_highres_boundary + 0.5 * lxref
# xc = np.array([math.fmod(xc_i, 1.0) for xc_i in xc])
# #or equivalently
xc = (left_corner_highres_boundary + 0.5 * lxref) % 1.0

print(f"our calculated centre: {xc}")

# hardcoded because I only want to understand my specific case (read from auriga_6_ics.conf_log.txt):
shift_unit = 2
ncoarse = 1 << 7  # == 2 ** 7 (x << y == x * 2 ** y)

desired_box_centre = np.array([0.5, 0.5, 0.5])

xshift = (desired_box_centre - xc) * ncoarse / shift_unit + desired_box_centre
xshift = xshift.astype(int) * shift_unit

print(f"our music shift: {xshift}")

adrian_shift = (desired_box_centre - xc * ncoarse).astype(np.int64)
print(f"adrians shift as noted in our music source: {adrian_shift}")
oliver_shift = np.array([-6, 0, -12])
assert (xshift == oliver_shift).all()

print(
    f"difference between adrians and olivers way to calculate the shift: {adrian_shift - oliver_shift}"
)

print(f"our music shift transformed to Mpc coordinates: {xshift / ncoarse * 100}")


xshift_in_mpc = np.array([-4.6875, 0, -9.375])
# according to music output:
highres_region_centre = np.array([0.566187, 0.517375, 0.612728]) * 100

shifted_highres_centre = highres_region_centre + xshift_in_mpc

print(
    f"our music shift applied to our calculated highresregion centre: {shifted_highres_centre}"
)
# print(f'adrians shift applied to our calculated highresregion centre: {shifted_highres_centre}')

print(
    f"difference between our calculated centre and the actual centre of the box: {desired_box_centre * 100 - shifted_highres_centre}"
)

h_adrian = 0.6777  # according to cosmICweb for the EAGLE sim

centre_adrian = np.array([56.50153741810241, 49.40761085700951, 49.634393647291695])
centre_music = np.array([58.25576087992683, 51.34632916228137, 51.68749302578122])

original_centre_in_adrians_zoom = np.array([4.367988, 1.307087, 7.743252]) / h_adrian

print(
    f"shift from original to his zooms according to adrian: {original_centre_in_adrians_zoom}"
)

print(f"shift from our centre to adrians: {centre_adrian - centre_music}")

original_centre_from_our_total_shift = xshift_in_mpc + (centre_adrian - centre_music)

# addition b/c they are in different directions, should cancel each other out
print(
    f"difference between adrians shift and our corrected shift: {original_centre_in_adrians_zoom + original_centre_from_our_total_shift}"
)
