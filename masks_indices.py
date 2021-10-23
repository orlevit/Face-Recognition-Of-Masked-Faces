import numpy as np
from config_file import config, FACE_MODEL_DENSITY, STRING_SIZE, SUNGLASSES_MASK_NAME, \
    COVID19_MASK_NAME, CENTER_FACE_PART, LEFT_FACE_PART, RIGHT_FACE_PART, LENS_RADIUS


def get_hat_mask_index(a1, b1, c1, x_left, x_right, x, y):
    """
    Extracting the indices for creating the hat mask.
    It is achieved by taking all the indices above parabolic line drawn between two points.
    :param a1: First coefficient of the parabolic equation.
    :param b1: Second coefficient of the parabolic equation.
    :param c1: Third coefficient of the parabolic equation.
    :param x_left: Left point.
    :param x_right: Right point.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :return: Hat mask indices.
    """
    index_list = []
    for i in range(len(x)):
        if (y[i] > (a1 * (x[i] ** 2) + b1 * x[i] + c1)) and (x[i] > x_left) and (x[i] < x_right):
            index_list.append(i)

    return index_list


def get_scarf_mask_index(a1, b1, c1, x_left, x_right, x, y):
    """
    Extracting the indices for creating the scarf mask.
    It is achieved by taking all the indices below parabolic line drawn between two points.
    :param a1: First coefficient of the parabolic equation.
    :param b1: Second coefficient of the parabolic equation.
    :param c1: Third coefficient of the parabolic equation.
    :param x_left: Left point.
    :param x_right: Right point.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :return: Scarf mask indices.
    """
    index_list = []
    for i in range(len(x)):
        if (y[i] < (a1 * x[i] ** 2 + b1 * x[i] + c1)) and (x[i] > x_left) and (x[i] < x_right):
            index_list.append(i)

    return index_list

#todo: change docstring
def get_eye_mask_index(a1, b1, c1, a2, b2, c2, x_left, x_right, x, y, z):
    """
    Extracting the indices for creating the eye mask.
    It is achieved by taking all the indices between two parabolic lines drawn between two points,
    and then taking the complement.

    :param a1: First coefficient of the first parabolic equation.
    :param b1: Second coefficient of the first parabolic equation.
    :param c1: Third coefficient of the first parabolic equation.
    :param a2: First coefficient of the second parabolic equation.
    :param b2: Second coefficient of the second parabolic equation.
    :param c2: Third coefficient of the second parabolic equation.
    :param x_left: Left point.
    :param x_right: Right point.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :param z: All the z of the points in the 3D model.

    :return: Eye mask indices.
    """
    index_list = []
    for i in range(len(x)):
        if ((y[i] < (a1 * (x[i] ** 2) + b1 * x[i] + c1)) and
                (y[i] > (a2 * x[i] ** 2 + b2 * x[i] + c2)) and
                (x[i] > x_left - 2) and (x[i] < x_right + 2)):
            index_list.append(i)

    # filtered_ind = [ii for ii, cord in enumerate(z) if (cord >= 0)]
    # index_list = np.setdiff1d(filtered_ind, index_list)

    return index_list


def make_eye_mask(x, y, z):
    """
    Getting the eye mask indices.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :param z: All the z of the points in the 3D model.

    :return: Eyes mask indices
    """
    x_left, y_left = x[config.eyemask.inds.left], y[config.eyemask.inds.left]
    x_right, y_right = x[config.eyemask.inds.right], y[config.eyemask.inds.right]
    x_middle, y_chosen_top = x[config.eyemask.inds.top], y[config.eyemask.inds.top]
    y_chosen_down = y[config.eyemask.inds.bottom]

    x_3points = [x_left, x_middle, x_right]
    y_3points1 = [y_left, y_chosen_down, y_right]
    y_3points2 = [y_left, y_chosen_top, y_right]

    a1, b1, c1 = np.polyfit(x_3points, y_3points2, 2)
    a2, b2, c2 = np.polyfit(x_3points, y_3points1, 2)

    index_list = get_eye_mask_index(a1, b1, c1, a2, b2, c2, x_left, x_right, x, y, z)

    return index_list


def make_scarf_mask(x, y):
    """
    Getting the scarf mask indices.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :return: Scarf mask indices
    """
    x_left, y_left = x[config.scarfmask.inds.left], y[config.scarfmask.inds.left]
    x_right, y_right = x[config.scarfmask.inds.right], y[config.scarfmask.inds.right]
    x_middle, y_chosen_top = x[config.scarfmask.inds.middle_top], y[config.scarfmask.inds.middle_top]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_chosen_top, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_scarf_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return index_list


def make_hat_mask(x, y):
    """
    Getting the hat mask indices.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :return: Hat mask indices
    """
    x_left, y_left = x[config.hatmask.inds.left], y[config.hatmask.inds.left]
    x_right, y_right = x[config.hatmask.inds.right], y[config.hatmask.inds.right]
    x_middle, y_chosen_down = x[config.hatmask.inds.middle_bottom], y[config.hatmask.inds.middle_bottom]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_chosen_down, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_hat_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return index_list


def make_covid19_mask(x, y, z):
    """
    Getting the covid19 mask indices.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :param z: All the z of the points in the 3D model.
    :return: Covid19 mask indices
    """
    center_middle = config.covid19mask.inds.center_middle
    right_middle = config.covid19mask.inds.right_middle
    left_lower = config.covid19mask.inds.left_lower
    right_lower = config.covid19mask.inds.right_lower
    left_upper_string1 = config.covid19mask.inds.left_upper_string1
    left_upper_string2 = config.covid19mask.inds.left_upper_string2
    left_lower_string1 = config.covid19mask.inds.left_lower_string1
    left_lower_string2 = config.covid19mask.inds.left_lower_string2
    right_upper_string1 = config.covid19mask.inds.right_upper_string1
    right_upper_string2 = config.covid19mask.inds.right_upper_string2
    right_lower_string1 = config.covid19mask.inds.right_lower_string1
    right_lower_string2 = config.covid19mask.inds.right_lower_string2

    covid19_mask_ind = center_face_ind(center_middle, right_middle, left_lower, right_lower, y, z)
    index_list1 = get_mask_string(left_upper_string1, left_upper_string2, LEFT_FACE_PART, x, y, z)
    index_list2 = get_mask_string(left_lower_string1, left_lower_string2, LEFT_FACE_PART, x, y, z)
    index_list3 = get_mask_string(right_upper_string1, right_upper_string2, RIGHT_FACE_PART, x, y, z)
    index_list4 = get_mask_string(right_lower_string1, right_lower_string2, RIGHT_FACE_PART, x, y, z)

    covid19_strings_ind = index_list1 + index_list2 + index_list3 + index_list4
    covid19_strings_ind = np.setdiff1d(covid19_strings_ind, covid19_mask_ind)

    return covid19_mask_ind, covid19_strings_ind


def make_sunglasses_mask(x, y):
    """
    Getting the sunglasses mask indices.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :return: Sunglasses mask indices
    """
    right_eye_ind = config[SUNGLASSES_MASK_NAME].inds.right
    left_eye_ind = get_sunglasses_left_eye(right_eye_ind, x, y)
    right_lens_inds, right_ind_right_eye, left_ind_right_eye = get_lens(right_eye_ind, x, y)
    left_lens_inds, right_ind_left_eye, left_ind_left_eye = get_lens(left_eye_ind, x, y)
    left_string = get_mask_string(left_ind_left_eye,
                                  config[COVID19_MASK_NAME].inds.left_upper_string2,
                                  CENTER_FACE_PART, x, x, y)
    center_string = get_mask_string(right_eye_ind, left_eye_ind, CENTER_FACE_PART, x, x, y)
    right_string = get_mask_string(config[COVID19_MASK_NAME].inds.right_upper_string2,
                                   right_ind_right_eye, CENTER_FACE_PART, x, x, y)
    sunglasses_lenses_inds = left_lens_inds + right_lens_inds
    sunglasses_strings_ind = left_string + center_string + right_string
    sunglasses_strings_ind = np.setdiff1d(sunglasses_strings_ind, sunglasses_lenses_inds)

    return sunglasses_lenses_inds, sunglasses_strings_ind


# TODO: Remove in final version
# This is a temp for testing new version of mask

def center_face_ind2(center_middle_ind, right_middle_ind, left_lower_ind, right_lower_ind, y, z):
    y_middle = y[[right_lower_ind, center_middle_ind, right_middle_ind]]
    z_middle = z[[right_lower_ind, center_middle_ind, right_middle_ind]]
    y_twist = y[[31919, 29207, 28040]]
    z_twist = z[[31919, 29207, 28040]]
    y_lower = y[[left_lower_ind, 31919]]
    z_lower = z[[left_lower_ind, 31919]]

    a1, b1, c1 = np.polyfit(y_middle, z_middle, 2)
    a2, b2, c2 = np.polyfit(y_twist, z_twist, 2)
    m, n = np.polyfit(y_lower, z_lower, 1)

    index_list = []
    for ii, y_i in enumerate(y):
        if (z[ii] >= (a1 * (y_i ** 2) + b1 * y_i + c1)) and (-0.1456907 < y[ii] < y[right_middle_ind]) or \
                (z[ii] >= (a2 * (y_i ** 2) + b2 * y_i + c2)) and (y[31919] < y[ii] < -0.1456907) or \
                (z[ii] >= (m * y_i + n)) and (y[left_lower_ind] < y[ii] < y[31919]):
            index_list.append(ii)

    return index_list


def center_face_ind(center_middle_ind, right_middle_ind, left_lower_ind, right_lower_ind, y, z):
    """
    Extracting the middle part of the covid19 mask(the part without the strings).
    It is achieved by creating a parabolic line from 3 known points on the mask and taking all the
    points above it and between a range, In addition the part below the chin is taken.
    :param center_middle_ind: Center point(First point for the parabola).
    :param right_middle_ind: Center point(Second point for the parabola).
    :param left_lower_ind: Left point(The end of the mask on lead - below the chin).
    :param right_lower_ind: Center point(Third point for the parabola).
    :param y: All the y of the points in the 3D model.
    :param z: All the z of the points in the 3D model.
    :return: The indices of the center part of the covid19 mask.
    """
    y_middle = y[[right_lower_ind, center_middle_ind, right_middle_ind]]
    z_middle = z[[right_lower_ind, center_middle_ind, right_middle_ind]]
    y_lower = y[[left_lower_ind, right_lower_ind]]
    z_lower = z[[left_lower_ind, right_lower_ind]]

    a, b, c = np.polyfit(y_middle, z_middle, 2)
    m, n = np.polyfit(y_lower, z_lower, 1)

    index_list = []
    for ii, y_i in enumerate(y):
        if (z[ii] >= (a * (y_i ** 2) + b * y_i + c)) and (y[right_lower_ind] < y[ii] < y[right_middle_ind]) or \
                (z[ii] >= (m * y_i + n)) and (y[left_lower_ind] < y[ii] < y[right_lower_ind]):
            index_list.append(ii)

    return index_list


# TODO: Remove in final version
# This is a temp for testing thinner string

def get_mask_string2(ind1, ind2, face_side, split_face_cord, x, y):
    if face_side == LEFT_FACE_PART:
        filtered_ind = [ii for ii, cord in enumerate(split_face_cord) if (cord >= 0)]
    elif face_side == RIGHT_FACE_PART:
        filtered_ind = [ii for ii, cord in enumerate(split_face_cord) if (cord < 0)]
    else:  # CENTER_FACE_PART
        filtered_ind = [ii for ii, cord in enumerate(split_face_cord)]

    x_pos = x[filtered_ind]
    y_pos = y[filtered_ind]

    # Draw line
    received_x_pt = [x[ind1], x[ind2]]
    received_y_pt = [y[ind1], y[ind2]]
    m, mb = np.polyfit(received_x_pt, received_y_pt, 1)

    start_x = min(received_x_pt)
    end_x = max(received_x_pt)
    line_list = []

    for i in np.arange(start_x, end_x + FACE_MODEL_DENSITY, FACE_MODEL_DENSITY):
        distances = np.asarray(((y_pos - (m * i + mb)) ** 2 + (x_pos - i) ** 2))
        string_size_ind = np.argpartition(distances, 1)[:1]
        cond_ind = list(np.asarray(filtered_ind)[string_size_ind])
        line_list += cond_ind

    return np.unique(np.asarray(line_list)).tolist()


def get_mask_string(ind1, ind2, face_side, split_face_cord, x, y):
    """
    Extracting the string part of the covid19 mask.
    :param ind1: First point of the string.
    :param ind2: Second point of the string.
    :param face_side: Whether it is the left or right side.
    :param split_face_cord: An enumerator for points indices.
    :param x: All the y of the points in the 3D model.
    :param y: All the z of the points in the 3D model.
    :return: The indices of the string part of the covid19 mask.
    """
    if face_side == LEFT_FACE_PART:
        filtered_ind = [ii for ii, cord in enumerate(split_face_cord) if (cord >= 0)]
    elif face_side == RIGHT_FACE_PART:
        filtered_ind = [ii for ii, cord in enumerate(split_face_cord) if (cord < 0)]
    else:  # CENTER_FACE_PART
        filtered_ind = [ii for ii, cord in enumerate(split_face_cord)]

    x_pos = x[filtered_ind]
    y_pos = y[filtered_ind]

    # Draw line
    received_x_pt = [x[ind1], x[ind2]]
    received_y_pt = [y[ind1], y[ind2]]
    m, mb = np.polyfit(received_x_pt, received_y_pt, 1)

    start_x = min(received_x_pt)
    end_x = max(received_x_pt)
    line_list = []

    for i in np.arange(start_x, end_x + FACE_MODEL_DENSITY, FACE_MODEL_DENSITY):
        distances = np.asarray(((y_pos - (m * i + mb)) ** 2 + (x_pos - i) ** 2))
        string_size_ind = np.argpartition(distances, STRING_SIZE)[:STRING_SIZE]
        cond_ind = list(np.asarray(filtered_ind)[string_size_ind])
        line_list += cond_ind

    return np.unique(np.asarray(line_list)).tolist()


def get_sunglasses_left_eye(eye_ind, x, y):
    """
    Getting the center point on the left eye out of the first selected point in the middle of the eye.
    This is done in order to get symmetry.
    :param eye_ind: The center of the lens - This is selected in advance.
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :return: The indices of center of the left lens mask.
    """
    distances = []
    indices = []

    for i in np.arange(len(x)):
        if -x[eye_ind] - LENS_RADIUS < x[i] < -x[eye_ind] + LENS_RADIUS and \
                y[eye_ind] - LENS_RADIUS < y[i] < y[eye_ind] + LENS_RADIUS:
            distances.append((-x[eye_ind] - x[i]) ** 2 + (y[eye_ind] - y[i]) ** 2)
            indices.append(i)

    array = np.column_stack((distances, indices))
    left_eye = array[np.argmin(array[:, 0]), 1].astype(int)
    return left_eye


def get_lens(center_eye_ind, x, y):
    """
    Getting one lens of the sunglasses mak.
    This is done by taking all points that are in certain radius from the center point.
    :param center_eye_ind: Whether this is the left or right lens
    :param x: All the x of the points in the 3D model.
    :param y: All the y of the points in the 3D model.
    :return: The indices of lens of the sunglasses mak.
    """
    distances = []
    indices = []

    for i in np.arange(len(x)):
        if ((x[center_eye_ind] - x[i]) ** 2 + (y[center_eye_ind] - y[i]) ** 2) <= LENS_RADIUS:
            distances.append(x[i])
            indices.append(i)

    array = np.column_stack((distances, indices))
    right_ind = array[np.argmin(array[:, 0]), 1].astype(int)
    left_ind = array[np.argmax(array[:, 0]), 1].astype(int)

    return indices, right_ind, left_ind
