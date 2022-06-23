# #Part 3: The return of the shift

import numpy as np

# necessary for Music's calculation
shift_unit = 2
ncoarse = 1 << 7  # == 2 ** 7 (x << y == x * 2 ** y)
desired_box_centre = np.array([0.5, 0.5, 0.5])


def music_shift(
    highres_centre: np.ndarray = None,
    left_corner: np.ndarray = None,
    right_corner: np.ndarray = None,
):
    if highres_centre is None and left_corner is None and right_corner is None:
        print("Please provide something from which we can calculate the shift.")
    if highres_centre is not None:
        xc = highres_centre
    if highres_centre is None and left_corner is not None and right_corner is not None:
        left_corner_highres_boundary = left_corner
        right_corner_highres_boundary = right_corner

        # this seems to be the extent of the highres region in units such that it lies in [0...1[
        lxref = right_corner_highres_boundary - left_corner_highres_boundary

        # the centre of the highres region in Music:
        xc = (left_corner_highres_boundary + 0.5 * lxref) % 1.0

    xshift = (desired_box_centre - xc) * ncoarse / shift_unit + desired_box_centre
    xshift = xshift.astype(int) * shift_unit

    return xshift


left_corner_music_ics = np.array([0.530964, 0.476099, 0.579666])
right_corner_music_ics = np.array([0.601410, 0.558650, 0.645789])
centre_music_ics = np.array([0.5661869, 0.51737463, 0.61272764])

# #Proudly presented by Trial & Error Productions
maximum_negative_boundary_for_same_shift = np.array([0.01149, 0.0251, 0.011])
maximum_positive_boundary_for_same_shift = np.array([0.004125, 0.006, 0.00425])

# #Secret algorithm of Trial & Error Productions
# print(f'Original music shift: {music_shift(centre_music_ics)}')
# print(f'Music shift for lower center: {music_shift(centre_music_ics - maximum_negative_boundary_for_same_shift)}')
# print(f'Music shift for upper center: {music_shift(centre_music_ics + maximum_positive_boundary_for_same_shift)}')

print(
    f"Lowest centre values for same Music shift: {(centre_music_ics - maximum_negative_boundary_for_same_shift) * 100}"
)
print(
    f"Highest centre values for same Music shift: {(centre_music_ics + maximum_positive_boundary_for_same_shift) * 100}"
)
