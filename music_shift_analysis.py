import numpy as np

# import math

# aka x0ref in music code:
left_corner_highres_boundary = np.array([0.530964, 0.476099, 0.579666])
# aka x1ref in music code:
right_corner_highres_boundary = np.array([0.601410, 0.558650, 0.645789])

# this seems to be the extent of the highres region in units such that it lies in [0...1[
lxref = right_corner_highres_boundary - left_corner_highres_boundary

# presumable the centre of the highres region:
# xc = left_corner_highres_boundary + 0.5 * lxref
# xc = np.array([math.fmod(xc_i, 1.0) for xc_i in xc])
# #or equivalently
xc = (left_corner_highres_boundary + 0.5 * lxref) % 1.0

print(xc)

# hardcoded because I only want to understand my specific case (read from auriga_6_ics.conf_log.txt):
shift_unit = 2
ncoarse = 1 << 7  # == 2 ** 7 (x << y == x * 2 ** y)

desired_box_centre = np.array([0.5, 0.5, 0.5])

xshift = (desired_box_centre - xc) * ncoarse / shift_unit + desired_box_centre
xshift = xshift.astype(int) * shift_unit

print(xshift)

adrian_shift = (desired_box_centre - xc * ncoarse).astype(np.int64)
print("a", adrian_shift)
oliver_shift = np.array([-6, 0, -12])
assert (xshift == oliver_shift).all()

print(adrian_shift - oliver_shift)

print(xshift / ncoarse * 100)


xshift_in_mpc = np.array([-4.6875, 0, -9.375])
#according to music output:
highres_region_centre = np.array([0.566187, 0.517375, 0.612728]) * 100

shifted_highres_centre = highres_region_centre + xshift_in_mpc

print(desired_box_centre * 100 - shifted_highres_centre)