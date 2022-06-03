import numpy as np
# import math

#aka x0ref in music code:
left_corner_highres_boundary = np.array([0.530964,0.476099,0.579666])
#aka x1ref in music code:
right_corner_highres_boundary = np.array([0.601410,0.558650,0.645789])

#this seems to be the extent of the highres region in units such that it lies in [0...1[
lxref = left_corner_highres_boundary - right_corner_highres_boundary

#presumable the centre of the highres region:
# xc = left_corner_highres_boundary + 0.5 * lxref  
# xc = np.array([math.fmod(xc_i, 1.0) for xc_i in xc])
# #or equivalently
xc = (left_corner_highres_boundary + 0.5 * lxref) % 1.0

print(xc)

#hardcoded because I only want to understand my specific case (read from auriga_6_ics.conf_log.txt):
shift_unit = 2
ncoarse = 7

normalised_box_centre = np.array([0.5, 0.5, 0.5])

xshift = (normalised_box_centre - xc) * ncoarse  + normalised_box_centre * shift_unit

print(xshift)