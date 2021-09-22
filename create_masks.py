import cv2
import numpy as np
import pandas as pd
from time import time
from skimage.filters import threshold_otsu
from project_on_image import transform_vertices
from helpers import scale
from masks_indices import make_eye_mask, make_hat_mask, make_corona_mask, \
    make_scarf_mask, make_sunglasses_mask
from config_file import config, VERTICES_PATH, EYE_MASK_NAME, HAT_MASK_NAME, SCARF_MASK_NAME, CORONA_MASK_NAME, \
    SUNGLASSES_MASK_NAME, NEAR_NEIGHBOUR_STRIDE, MIN_MASK_SIZE, FILTER_MASK_RIGHT_POINT_IMAGE_SIZE, SAME_AREA_DIST, \
    FILTER_SIZE_MASK_RIGHT_POINT, FILTER_SIZE_MASK_ADD_LEFT_POINT, FILTER_SIZE_MASK_ADD_RIGHT_POINT, THRESHOLD_BUFFER
from line_profiler_pycharm import profile


@profile
def render(img, r_img, pose, mask_name, scale_factor):
    # Transform the 3DMM according to the pose and get only frontal face areas
    frontal_mask, frontal_add_mask, frontal_rest = get_frontal(r_img, pose, mask_name, scale_factor)

    # Whether to add the forehead to the mask, this is currently only used for eye and hat masks
    if config[mask_name].add_forehead:
        mask_x, mask_y = add_forehead_mask(img, pose)
        mask_x, mask_y = np.append(mask_x, frontal_mask[:, 0]), np.append(mask_y, frontal_mask[:, 1])
    else:
        mask_x, mask_y = frontal_mask[:, 0], frontal_mask[:, 1]

    mask_add_x, mask_add_y = frontal_add_mask[:, 0], frontal_add_mask[:, 1]

    # Perform morphological close
    morph_mask_main_x, morph_mask_main_y = morphological_op(mask_x, mask_y, img, config[mask_name].filter_size)
    morph_mask_add_x, morph_mask_add_y = morphological_op(mask_add_x, mask_add_y, img,
                                                          FILTER_SIZE_MASK_ADD_LEFT_POINT,
                                                          FILTER_SIZE_MASK_ADD_RIGHT_POINT, cv2.MORPH_DILATE)
    morph_mask_x = np.append(morph_mask_main_x, morph_mask_add_x)
    morph_mask_y = np.append(morph_mask_main_y, morph_mask_add_y)

    if config[mask_name].draw_rest_mask:
        rest_x, rest_y = frontal_rest[:, 0], frontal_rest[:, 1]
        morph_rest_x, morph_rest_y = morphological_op(rest_x, rest_y, img, config[mask_name].filter_size)
    else:
        morph_rest_x, morph_rest_y = None, None

    return morph_mask_x, morph_mask_y, morph_rest_x, morph_rest_y


@profile
def neighbors_cells_z(mask_on_img, x_pixel, y_pixel, max_x, max_y):
    x_right_limit = min(x_pixel + NEAR_NEIGHBOUR_STRIDE, max_x)
    y_upper_limit = min(y_pixel + NEAR_NEIGHBOUR_STRIDE, max_y)
    x_left_limit = max(x_pixel - NEAR_NEIGHBOUR_STRIDE, 0)
    y_bottom_limit = max(y_pixel - NEAR_NEIGHBOUR_STRIDE, 0)

    z_neighbors = []

    for x in range(x_left_limit, x_right_limit + 1):
        for y in range(y_bottom_limit, y_upper_limit + 1):
            if not isinstance(mask_on_img[y, x], type(None)):
                z_neighbors.extend(mask_on_img[y, x])

    return z_neighbors


@profile
def clustering(elements_list):
    threshold = threshold_otsu(np.array(elements_list))
    cluster1_arr = []
    cluster2_arr = []
    equal_threshold = []
    for i in elements_list:
        if i < threshold:
            cluster1_arr.append(i)
        elif i > threshold:
            cluster2_arr.append(i)
        else:
            equal_threshold.append(i)

    if not cluster1_arr:
        cluster1_arr.extend(equal_threshold)
    else:
        cluster2_arr.extend(equal_threshold)

    cluster1 = np.mean(cluster1_arr)
    cluster2 = np.mean(cluster2_arr)

    return cluster1, cluster2


@profile
def threshold_front(r_img, df, frontal_mask_all):
    img_x_dim, img_y_dim = r_img.shape[1], r_img.shape[0]
    mask_on_img = np.asarray([[None] * img_x_dim] * img_y_dim)
    mask_on_img_front = np.zeros((img_y_dim, img_x_dim))

    # Each pixel contains a list of all the Z coordinates from the 3D model
    for x, y, z in zip(df.x, df.y, df.z):
        if isinstance(mask_on_img[y, x], type(None)):
            mask_on_img[y, x] = [z]
        else:
            mask_on_img[y, x].append(z)

    for x, y, z in zip(frontal_mask_all.x, frontal_mask_all.y, frontal_mask_all.z):
        surrounding_mask = neighbors_cells_z(mask_on_img, x, y, img_x_dim - 1, img_y_dim - 1)
        mask_on_img_front[y, x] = 1

        if len(np.unique(surrounding_mask)) not in [0, 1]:
            cluster1, cluster2 = clustering(surrounding_mask)
            diff = abs(cluster1 - cluster2)
            if SAME_AREA_DIST < diff:
                min_cluster = min(cluster1, cluster2)
                threshold_buffer = diff * THRESHOLD_BUFFER
                threshold = min_cluster + threshold_buffer
                mask_on_img_front[y, x] = 0 if z < threshold else 1

    mask_marks = np.asarray(np.where(mask_on_img_front == 1)).T[:, [1, 0]]

    return mask_marks


# TODO: project  images without added strings on image
@profile
def get_frontal(r_img, pose, mask_name, scale_factor):
    # Not working with "config[mask_name].mask_add_ind = None" !!!

    # An indication whether it is a mask coordinate, additional  mask or rest of the head and add them to the matrices
    mask_marks = 3 * np.ones([config[mask_name].mask_ind.shape[0], 1], dtype=bool)
    mask_add_marks = 2 * np.ones([config[mask_name].mask_add_ind.shape[0], 1], dtype=bool)
    rest_marks = np.ones([config[mask_name].rest_ind.shape[0], 1], dtype=bool)
    mask_stacked = np.hstack((config[mask_name].mask_ind, mask_marks))
    mask_add_stacked = np.hstack((config[mask_name].mask_add_ind, mask_add_marks))
    rest_stacked = np.hstack((config[mask_name].rest_ind, rest_marks))
    combined_float = np.vstack((mask_stacked, mask_add_stacked, rest_stacked))

    # Masks projection on the image plane
    combined_float[:, :3] = transform_vertices(r_img, pose, combined_float[:, :3])

    # turn values from float to integer
    combined = np.round(combined_float).astype(int)
    df = pd.DataFrame(combined, columns=['x', 'y', 'z', 'mask'])
    df = df[((0 <= df.x) & (df.x <= r_img.shape[1] - 1)) & ((0 <= df.y) & (df.y <= r_img.shape[0] - 1))]

    # Order the coordinates by z, remove duplicates x,y,mask values and keep the first occurrence
    # Only the closer z pixels is consider as a candidate for appearing in that pixel
    unique_df = df.sort_values(['z'], ascending=False).drop_duplicates(['x', 'y', 'mask'], keep='first')

    frontal_main_mask_with_bg = unique_df[(unique_df['mask'] == 3)][['x', 'y', 'z']]
    frontal_add_mask_with_bg = unique_df[(unique_df['mask'] == 2)][['x', 'y', 'z']]
    frontal_rest_mask_with_bg = unique_df[(unique_df['mask'] == 1)][['x', 'y', 'z']]

    # Check each point if it is came from frontal or hidden area of tha face
    famwb_arr = threshold_front(r_img, df, frontal_add_mask_with_bg)
    # TODO: Switch comments to take frontal mask center if NOT multithread!
    ############## Switch comments to take frontal mask center if NOT multithread! #######################################
    # main_mask_on_image = threshold_front(img, df, frontal_main_mask_with_bg)
    # main_mask_on_image = mark_image_with_mask(img, frontal_main_mask_with_bg.x, frontal_main_mask_with_bg.y)
    fmmwb_arr = frontal_main_mask_with_bg[['x', 'y']].to_numpy()
    #####################################################################################################################
    # rest_mask_on_image = mark_image_with_mask(img, frontal_rest_mask_with_bg.x, frontal_rest_mask_with_bg.y)

    if config[mask_name].draw_rest_mask:
        frmwb_arr = frontal_rest_mask_with_bg[['x', 'y']].to_numpy()
    else:
        frmwb_arr = []

    frontal_mask, frontal_add_mask, frontal_rest = scale(r_img, fmmwb_arr, famwb_arr, frmwb_arr, scale_factor)

    return frontal_mask, frontal_add_mask, frontal_rest


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
    # Get the average color of the whole mask
    image_bg = image.copy()
    image_bg_effective_size = image_bg.shape[0] * image_bg.shape[1] - len(mask_x)
    image_bg[mask_y.astype(int), mask_x.astype(int), :] = [0, 0, 0]
    image_bg_blue = image_bg[:, :, 0]
    image_bg_green = image_bg[:, :, 1]
    image_bg_red = image_bg[:, :, 2]
    image_bg_blue_val = np.sum(image_bg_blue) / image_bg_effective_size
    image_bg_green_val = np.sum(image_bg_green) / image_bg_effective_size
    image_bg_red_val = np.sum(image_bg_red) / image_bg_effective_size

    return [image_bg_blue_val, image_bg_green_val, image_bg_red_val]


def calc_filter_size(mask_x, mask_y, left_filter_size, right_filter_dim):
    min_x, max_x = np.min(mask_x), np.max(mask_x)
    min_y, max_y = np.min(mask_y), np.max(mask_y)
    x_diff, y_diff = max_x - min_x, max_y - min_y
    mask_size = max(x_diff, y_diff)

    x = [MIN_MASK_SIZE, FILTER_MASK_RIGHT_POINT_IMAGE_SIZE]
    y = [np.mean(left_filter_size), right_filter_dim]

    a, b = np.polyfit(x, y, 1)
    filter_dim = int(np.ceil(a * mask_size + b))
    if filter_dim < 1:
        filter_dim = 1
    filter_size = (filter_dim, filter_dim)

    return filter_size


def morphological_op(mask_x, mask_y, image, left_filter_size=config[EYE_MASK_NAME].filter_size,
                     right_filter_dim=FILTER_SIZE_MASK_RIGHT_POINT, morph_op=cv2.MORPH_CLOSE):
    mask_on_image = np.zeros_like(image)
    for x, y in zip(mask_x, mask_y):
        mask_on_image[y, x, :] = [255, 255, 255]

    # morphology close
    gray_mask = cv2.cvtColor(mask_on_image, cv2.COLOR_BGR2GRAY)
    res, thresh_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
    filter_size = calc_filter_size(mask_x, mask_y, left_filter_size, right_filter_dim)
    kernel = np.ones(filter_size, np.uint8)  # kernel filter
    morph_mask = cv2.morphologyEx(thresh_mask, morph_op, kernel)
    yy, xx = np.where(morph_mask == 255)

    return xx, yy


def add_forehead_mask(image, pose):  # , mask_trans_vertices, rest_trans_vertices):
    frontal_mask, frontal_add_mask, frontal_rest = get_frontal(image, pose, HAT_MASK_NAME)

    # Perform morphological close
    morph_mask_x, morph_mask_y = morphological_op(frontal_mask[:, 0], frontal_mask[:, 1],
                                                  image, config[HAT_MASK_NAME].filter_size)

    mask_on_img = np.zeros_like(image)
    mask_x_ind, mask_y_ind = morph_mask_x, morph_mask_y
    for x, y in zip(mask_x_ind, mask_y_ind):
        mask_on_img[y, x] = 255

    bottom_hat = np.empty(image.shape[1])
    bottom_hat[:] = np.nan
    for i in range(image.shape[1]):
        iy, _ = np.where(mask_on_img[:, i])
        if iy.any():
            bottom_hat[i] = np.min(iy)

    all_face_proj = np.concatenate((frontal_mask, frontal_rest), axis=0)
    all_face_proj_y = all_face_proj[:, 1]
    max_proj_y, min_proj_y = np.max(all_face_proj_y), np.min(all_face_proj_y)
    kernel_len = int((max_proj_y - min_proj_y) / 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, kernel_len))
    dilated_image = cv2.dilate(mask_on_img, kernel, iterations=1)

    forehead_x_ind, forehead_y_ind = [], []
    for i in range(image.shape[1]):
        iy, _ = np.where(dilated_image[:, i])
        jj = np.where(iy <= bottom_hat[i])
        if jj[0].any():
            j = iy[jj]
            forehead_x_ind.extend([i] * len(j))
            forehead_y_ind.extend(j)

    return np.asarray(forehead_x_ind), np.asarray(forehead_y_ind)


def get_rest_mask(mask_ind, mask_add_ind, vertices):
    rest_of_head_ind_with_add = np.setdiff1d(range(vertices.shape[0]), mask_ind)
    rest_of_head_ind = np.setdiff1d(rest_of_head_ind_with_add, mask_add_ind)
    rest_of_head_mask = index_on_vertices(rest_of_head_ind, vertices)

    return rest_of_head_mask


def load_3dmm():
    vertices = np.load(VERTICES_PATH)
    th = np.pi
    rotation_matrix = [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
    vertices_rotated = vertices.copy()
    vertices_rotated = np.matmul(vertices_rotated, rotation_matrix)
    return vertices, vertices_rotated


def masks_templates(masks_name):
    vertices, vertices_rotated = load_3dmm()
    x, y, z = vertices_rotated[:, 0], vertices_rotated[:, 1], vertices_rotated[:, 2]
    eye_mask_ind = make_eye_mask(x, y)
    hat_mask_ind = make_hat_mask(x, y)
    scarf_mask_ind = make_scarf_mask(x, y)
    corona_mask_ind, add_corona_ind = make_corona_mask(x, y, z)
    sunglasses_mask_ind, add_sunglasses_ind = make_sunglasses_mask(x, y)

    masks_order = [EYE_MASK_NAME, HAT_MASK_NAME, SCARF_MASK_NAME, CORONA_MASK_NAME, SUNGLASSES_MASK_NAME]
    masks_ind = [eye_mask_ind, hat_mask_ind, scarf_mask_ind, corona_mask_ind, sunglasses_mask_ind]
    masks_add_ind = [None, None, None, add_corona_ind, add_sunglasses_ind]

    masks = [index_on_vertices(maskInd, vertices) for maskInd in masks_ind]
    masks_add = [None if mask_add_ind is None else
                 index_on_vertices(mask_add_ind, vertices) for mask_add_ind in masks_add_ind]
    rest_of_heads = [get_rest_mask(maskInd, maskAInd, vertices) for maskInd, maskAInd in zip(masks_ind, masks_add_ind)]
    masks_to_create = parse_masks_name(masks_name)
    add_mask_to_config(masks, masks_add, rest_of_heads, masks_order, masks_to_create)

    return masks_to_create


def parse_masks_name(masks_name):
    masks_names_parsed = masks_name.split(',')
    additional_mask = [config[name].additional_masks_req for name in masks_names_parsed
                       if config[name].additional_masks_req is not None]
    masks_names_parsed += additional_mask

    return np.unique(np.asarray(masks_names_parsed)).tolist()


def add_mask_to_config(masks, masks_add, rest_of_heads, masks_order, masks_to_create):
    for mask_name in masks_to_create:
        config[mask_name].mask_ind = masks[masks_order.index(mask_name)]
        config[mask_name].mask_add_ind = masks_add[masks_order.index(mask_name)]
        config[mask_name].rest_ind = rest_of_heads[masks_order.index(mask_name)]
