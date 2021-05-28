import cv2
import numpy as np

from config_file import config, VERTICES_PATH, FACE_MODEL_DENSITY, STRING_SIZE, \
    MORPHOLOGICAL_CLOSE_FILTER, EYE_MASK_NAME, HAT_MASK_NAME, SCARF_MASK_NAME, CORONA_MASK_NAME
from project_on_image import transform_vertices


# This is the opposite of the function in expression-net-old
def get_hat_mask_index(a1, b1, c1, x_left, x_right, x, y):
    index_list = []
    for i in range(len(x)):
        if ((y[i] > (a1 * (x[i] ** 2) + b1 * x[i] + c1)) and
                (x[i] > x_left) and (x[i] < x_right)):
            index_list.append(i)

    return index_list


# This is the opposite of the function in expression-net-old
# get the scarf mask indexes on the model
def get_scarf_mask_index(a1, b1, c1, x_left, x_right, x, y):
    index_list = []
    for i in range(len(x)):
        if (y[i] < (a1 * x[i] ** 2 + b1 * x[i] + c1)) and (x[i] > x_left) and (x[i] < x_right):
            index_list.append(i)

    return index_list


# This is the opposite of the function in expression-net-old
def get_eyes_mask_index(a1, b1, c1, a2, b2, c2, x_left, x_right, x, y):
    index_list = []
    for i in range(len(x)):
        if ((y[i] < (a1 * (x[i] ** 2) + b1 * x[i] + c1)) and
                (y[i] > (a2 * x[i] ** 2 + b2 * x[i] + c2)) and
                (x[i] > x_left - 2) and (x[i] < x_right + 2)):
            index_list.append(i)

    index_list = np.setdiff1d(range(len(x)), index_list)
    return index_list


def make_eye_mask(x, y):
    x_left, y_left = x[config.eyemask.inds.left], y[config.eyemask.inds.left]
    x_right, y_right = x[config.eyemask.inds.right], y[config.eyemask.inds.right]
    x_middle, y_chosen_top = x[config.eyemask.inds.top], y[config.eyemask.inds.top]
    y_chosen_down = y[config.eyemask.inds.bottom]

    x_3points = [x_left, x_middle, x_right]
    y_3points1 = [y_left, y_chosen_down, y_right]
    y_3points2 = [y_left, y_chosen_top, y_right]

    a1, b1, c1 = np.polyfit(x_3points, y_3points2, 2)
    a2, b2, c2 = np.polyfit(x_3points, y_3points1, 2)

    index_list = get_eyes_mask_index(a1, b1, c1, a2, b2, c2, x_left, x_right, x, y)

    return index_list


def make_scarf_mask(x, y):
    x_left, y_left = x[config.scarfmask.inds.left], y[config.scarfmask.inds.left]
    x_right, y_right = x[config.scarfmask.inds.right], y[config.scarfmask.inds.right]
    x_middle, y_chosen_top = x[config.scarfmask.inds.middle_top], y[config.scarfmask.inds.middle_top]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_chosen_top, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_scarf_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return index_list


# create hat mask
def make_hat_mask(x, y):
    x_left, y_left = x[config.hatmask.inds.left], y[config.hatmask.inds.left]
    x_right, y_right = x[config.hatmask.inds.right], y[config.hatmask.inds.right]
    x_middle, y_chosen_down = x[config.hatmask.inds.middle_bottom], y[config.hatmask.inds.middle_bottom]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_chosen_down, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_hat_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return index_list


def make_corona_mask(x, y, z):
    center_middle = config.coronamask.inds.center_middle
    right_middle = config.coronamask.inds.right_middle
    left_lower = config.coronamask.inds.left_lower
    right_lower = config.coronamask.inds.right_lower
    left_upper_string1 = config.coronamask.inds.left_upper_string1
    left_upper_string2 = config.coronamask.inds.left_upper_string2
    left_lower_string1 = config.coronamask.inds.left_lower_string1
    left_lower_string2 = config.coronamask.inds.left_lower_string2
    right_upper_string1 = config.coronamask.inds.right_upper_string1
    right_upper_string2 = config.coronamask.inds.right_upper_string2
    right_lower_string1 = config.coronamask.inds.right_lower_string1
    right_lower_string2 = config.coronamask.inds.right_lower_string2

    corona_mask_ind = center_face_ind(center_middle, right_middle, left_lower, right_lower, y, z)
    index_list1 = get_mask_string(left_upper_string1, left_upper_string2, 'LEFT', x, y, z)
    index_list2 = get_mask_string(left_lower_string1, left_lower_string2, 'LEFT', x, y, z)
    index_list3 = get_mask_string(right_upper_string1, right_upper_string2, 'RIGHT', x, y, z)
    index_list4 = get_mask_string(right_lower_string1, right_lower_string2, 'RIGHT', x, y, z)

    corona_strings_ind = index_list1 + index_list2 + index_list3 + index_list4

    return corona_mask_ind, corona_strings_ind


def center_face_ind(center_middle_ind, right_middle_ind, left_lower_ind, right_lower_ind, y, z):
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


def get_mask_string(ind1, ind2, face_side, x, y, z):
    if face_side == 'LEFT':
        filtered_ind = [ii for ii, x_i in enumerate(x) if (x_i >= 0)]
    else:  # right face
        filtered_ind = [ii for ii, x_i in enumerate(x) if (x_i < 0)]

    y_pos = y[filtered_ind]
    z_pos = z[filtered_ind]

    # Draw line
    received_y_pt = [y[ind1], y[ind2]]
    received_z_pt = [z[ind1], z[ind2]]
    m, mb = np.polyfit(received_y_pt, received_z_pt, 1)

    start_y = min(received_y_pt)
    end_y = max(received_y_pt)
    line_list = []
    for i in np.arange(start_y, end_y + FACE_MODEL_DENSITY, FACE_MODEL_DENSITY):
        distances = np.asarray(((z_pos - (m * i + mb)) ** 2 + (y_pos - i) ** 2))
        string_size_ind = np.argpartition(distances, STRING_SIZE)[:STRING_SIZE]
        cond_ind = list(np.asarray(filtered_ind)[string_size_ind])
        line_list += cond_ind

    return np.unique(np.asarray(line_list)).tolist()


def render(img, pose, mask_name):
    # Transform the 3DMM according to the pose
    mask_trans_vertices = transform_vertices(img, pose, config[mask_name].mask_ind)
    rest_trans_vertices = transform_vertices(img, pose, config[mask_name].rest_ind)

    # Whether to add the forehead to the mask, this is currently only used for eye and hat masks
    if config[mask_name].add_forehead:
        mask_x, mask_y = add_forehead_mask(img, pose)
        mask_x, mask_y = np.append(mask_x.flatten(), mask_trans_vertices[:, 0]), \
                         np.append(mask_y.flatten(), mask_trans_vertices[:, 1])
    else:
        mask_x, mask_y = mask_trans_vertices[:, 0], mask_trans_vertices[:, 1]

    rest_x, rest_y = rest_trans_vertices[:, 0], rest_trans_vertices[:, 1]

    # Perform morphological close
    morph_mask_x, morph_mask_y = morphological_close(mask_x, mask_y, img)
    morph_rest_x, morph_rest_y = morphological_close(rest_x, rest_y, img)

    if config[mask_name].mask_add_ind is not None:
        mask_add_trans_vertices = np.round(transform_vertices(img, pose, config[mask_name].mask_add_ind)).astype(int)
        morph_mask_x = np.append(morph_mask_x, mask_add_trans_vertices[:, 0])
        morph_mask_y = np.append(morph_mask_y, mask_add_trans_vertices[:, 1])

    return morph_mask_x, morph_mask_y, morph_rest_x, morph_rest_y


def index_on_vertices(index_list, vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    x_mask = x[index_list]
    y_mask = y[index_list]
    z_mask = z[index_list]
    size_array = (x_mask.shape[0], 3)
    mask = np.zeros(size_array)
    mask[:, 0] = x_mask
    mask[:, 1] = y_mask
    mask[:, 2] = z_mask

    return mask


def bg_color(mask_x, mask_y, image):
    morph_mask_x, morph_mask_y = morphological_close(mask_x, mask_y, image)
    # Get the average color of the whole mask
    image_bg = image.copy()
    image_bg_effective_size = image_bg.shape[0] * image_bg.shape[1] - len(mask_x)
    image_bg[morph_mask_x.astype(int), morph_mask_y.astype(int), :] = [0, 0, 0]
    image_bg_blue = image_bg[:, :, 0]
    image_bg_green = image_bg[:, :, 1]
    image_bg_red = image_bg[:, :, 2]
    image_bg_blue_val = np.sum(image_bg_blue) / image_bg_effective_size
    image_bg_green_val = np.sum(image_bg_green) / image_bg_effective_size
    image_bg_red_val = np.sum(image_bg_red) / image_bg_effective_size

    return [image_bg_blue_val, image_bg_green_val, image_bg_red_val]


def morphological_close(mask_x, mask_y, image):
    mask_on_image = np.zeros_like(image)
    for x, y in zip(mask_x, mask_y):
        if (0 <= x <= mask_on_image.shape[0] - 1) and (0 <= y <= mask_on_image.shape[1] - 1):
            mask_on_image[int(y), int(x), :] = [255, 255, 255]

    # morphology close
    gray_mask = cv2.cvtColor(mask_on_image, cv2.COLOR_BGR2GRAY)
    res, thresh_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones(MORPHOLOGICAL_CLOSE_FILTER, np.uint8)  # kernel filter
    morph_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)
    yy, xx = np.where(morph_mask == 255)

    return xx, yy


def add_forehead_mask(image, pose):  # , mask_trans_vertices, rest_trans_vertices):
    mask_trans_vertices = transform_vertices(image, pose, config[HAT_MASK_NAME].mask_ind)
    rest_trans_vertices = transform_vertices(image, pose, config[HAT_MASK_NAME].rest_ind)

    mask_on_img = np.zeros_like(image)
    mask_x_ind, mask_y_ind = mask_trans_vertices[:, 0].astype(int), mask_trans_vertices[:, 1].astype(int)
    for x, y in zip(mask_x_ind, mask_y_ind):
        mask_on_img[y, x] = 255

    bottom_hat = np.empty(image.shape[0])
    bottom_hat[:] = np.nan
    for i in range(image.shape[0]):
        iy, _ = np.where(mask_on_img[:, i])
        if iy.any():
            bottom_hat[i] = np.min(iy)

    all_face_proj = np.concatenate((mask_trans_vertices, rest_trans_vertices), axis=0)
    all_face_proj_y = all_face_proj[:, 1]
    max_proj_y, min_proj_y = np.max(all_face_proj_y), np.min(all_face_proj_y)
    kernel_len = int((max_proj_y - min_proj_y) / 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, kernel_len))
    dilated_image = cv2.dilate(mask_on_img, kernel, iterations=1)

    forehead_x_ind, forehead_y_ind = [], []
    for i in range(image.shape[0]):
        iy, _ = np.where(dilated_image[:, i])
        jj = np.where(iy <= bottom_hat[i])
        if jj[0].any():
            j = iy[jj]
            forehead_x_ind.append([i] * len(j))
            forehead_y_ind.append(j)

    return np.asarray(forehead_x_ind), np.asarray(forehead_y_ind)


def get_rest_mask(mask_ind, vertices):
    rest_of_head_ind = np.setdiff1d(range(vertices.shape[0]), mask_ind)
    rest_of_head_mask = index_on_vertices(rest_of_head_ind, vertices)

    return rest_of_head_mask


def load_3dmm():
    vertices = np.load(VERTICES_PATH)
    th = np.pi
    rotation_matrix = [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
    vertices_rotated = vertices.copy()
    vertices_rotated = np.matmul(vertices_rotated, rotation_matrix)
    return vertices, vertices_rotated


def create_masks(masks_name):
    vertices, vertices_rotated = load_3dmm()
    x, y, z = vertices_rotated[:, 0], vertices_rotated[:, 1], vertices_rotated[:, 2]
    eye_mask_ind = make_eye_mask(x, y)
    hat_mask_ind = make_hat_mask(x, y)
    scarf_mask_ind = make_scarf_mask(x, y)
    corona_mask_ind, add_corona_ind = make_corona_mask(x, y, z)

    masks_order = [EYE_MASK_NAME, HAT_MASK_NAME, SCARF_MASK_NAME, CORONA_MASK_NAME]
    masks_ind = [eye_mask_ind, hat_mask_ind, scarf_mask_ind, corona_mask_ind]
    masks_add_ind = [None, None, None, add_corona_ind]

    masks = [index_on_vertices(maskInd, vertices) for maskInd in masks_ind]
    masks_add = [None if mask_add_ind is None else \
                     index_on_vertices(mask_add_ind, vertices) for mask_add_ind in masks_add_ind]
    rest_of_heads = [get_rest_mask(maskInd, vertices) for maskInd in masks_ind]
    masks_to_create = parse_masks_name(masks_name)
    add_mask_to_config(masks, masks_add, rest_of_heads, masks_order, masks_to_create)

    return masks_to_create


def parse_masks_name(masks_name):
    masks_names_parsed = masks_name.split(',')
    additional_mask = [config[name].additional_masks_req for name in masks_names_parsed \
                       if config[name].additional_masks_req is not None]
    masks_names_parsed += additional_mask

    return np.unique(np.asarray(masks_names_parsed)).tolist()


def add_mask_to_config(masks, masks_add, rest_of_heads, masks_order, masks_to_create):
    for mask_name in masks_to_create:
        config[mask_name].mask_ind = masks[masks_order.index(mask_name)]
        config[mask_name].mask_add_ind = masks_add[masks_order.index(mask_name)]
        config[mask_name].rest_ind = rest_of_heads[masks_order.index(mask_name)]
