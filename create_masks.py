import cv2
from time import time ###############33
import numpy as np
from skimage.filters import threshold_multiotsu
from masks_indices import make_eye_mask, make_hat_mask, make_covid19_mask, make_scarf_mask, make_sunglasses_mask
from helpers import scale, split_head_mask_parts, get_1id_pose, resize_image, project_3d, color_face_mask, save_image, \
    head3d_z_dist
from config_file import config, VERTICES_PATH, EYE_MASK_NAME, HAT_MASK_NAME, SCARF_MASK_NAME, COVID19_MASK_NAME, \
    SUNGLASSES_MASK_NAME, NEAR_NEIGHBOUR_STRIDE, MIN_MASK_SIZE, FILTER_MASK_RIGHT_POINT_IMAGE_SIZE, \
    FILTER_SIZE_MASK_RIGHT_POINT, FILTER_SIZE_MASK_ADD_LEFT_POINT, FILTER_SIZE_MASK_ADD_RIGHT_POINT, THRESHOLD_BUFFER, \
    RANGE_CHECK, HEAD_3D_NAME

from line_profiler_pycharm import profile


@profile
def render(img, r_img, df_3dh, h3d2i, mask_name, scale_factor):
    # Get only frontal face areas
    frontal_mask, frontal_add_mask, frontal_rest = get_frontal(r_img, df_3dh, h3d2i, mask_name, scale_factor)

    # Whether to add the forehead to the mask, this is currently only used for eye and hat masks
    if config[mask_name].add_forehead:
        if mask_name == HAT_MASK_NAME:
            forehead_x, forehead_y = add_forehead_mask(frontal_mask, frontal_rest, img)
            config[HAT_MASK_NAME].forehead_x, config[HAT_MASK_NAME].forehead_y = forehead_x, forehead_y
        else:
            forehead_x, forehead_y = config[HAT_MASK_NAME].forehead_x, config[HAT_MASK_NAME].forehead_y
    else:
        forehead_x, forehead_y = [], []

    mask_x = np.append(forehead_x, frontal_mask[:, 0]).astype(int)
    mask_y = np.append(forehead_y, frontal_mask[:, 1]).astype(int)
    morph_mask_x, morph_mask_y = morphological_op(mask_x, mask_y, img, config[mask_name].filter_size)

    if config[mask_name].mask_add_ind is not None:
        mask_add_x, mask_add_y = frontal_add_mask[:, 0], frontal_add_mask[:, 1]
        morph_mask_add_x, morph_mask_add_y = morphological_op(mask_add_x, mask_add_y, img,
                                                              FILTER_SIZE_MASK_ADD_LEFT_POINT,
                                                              FILTER_SIZE_MASK_ADD_RIGHT_POINT, cv2.MORPH_DILATE)
        morph_mask_x = np.append(morph_mask_x, morph_mask_add_x)
        morph_mask_y = np.append(morph_mask_y, morph_mask_add_y)

    if config[mask_name].draw_rest_mask:
        rest_x, rest_y = frontal_rest[:, 0], frontal_rest[:, 1]
        morph_rest_x, morph_rest_y = morphological_op(rest_x, rest_y, img,
                                                      config[mask_name].filter_size, morph_op=cv2.MORPH_DILATE)
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
            if mask_on_img[y, x] is not None:
                z_neighbors.extend(mask_on_img[y, x])

    return np.asarray(z_neighbors, dtype=np.float)


@profile
def three_clusters_arrays(elements, thresholds, bin_half_size):
    is_bin2_free = True
    left_bin1_range = elements <= thresholds[0] + bin_half_size
    right_bin2_range = thresholds[1] - bin_half_size <= elements
    bin1 = elements[left_bin1_range][thresholds[0] - bin_half_size <= elements[left_bin1_range]]
    cluster1_arr = elements[~(left_bin1_range | right_bin2_range)]
    bin2_range = elements[right_bin2_range] <= thresholds[1] + bin_half_size
    bin2 = elements[right_bin2_range][bin2_range]
    cluster2_arr = elements[right_bin2_range][~bin2_range]

    if not cluster2_arr.size:
        cluster2_arr = np.append(cluster2_arr, bin2)
        is_bin2_free = False
        if not cluster1_arr.size:
            cluster1_arr = np.append(cluster1_arr, bin1)

    if not cluster1_arr.size:
        if bin2.size and is_bin2_free:
            cluster1_arr = np.append(cluster1_arr, bin2)
        else:
            cluster1_arr = np.append(cluster1_arr, bin1)

    return cluster1_arr, cluster2_arr


@profile
def two_clusters_arrays(elements, threshold, bin_half_size):
    less_range = elements < threshold - bin_half_size
    bigger_range = threshold + bin_half_size < elements
    cluster1_arr = elements[less_range]
    cluster2_arr = elements[bigger_range]
    bin1 = elements[~(less_range | bigger_range)]

    if not cluster1_arr.size:
        cluster1_arr = np.append(cluster1_arr, bin1)
    elif not cluster2_arr.size:
        cluster2_arr = np.append(cluster2_arr, bin1)

    return cluster1_arr, cluster2_arr


@profile
def otsu_clustering(elements, cluster_number, bins_number, bin_half_size):
    thresholds = threshold_multiotsu(elements, cluster_number, nbins=bins_number)

    if cluster_number == 2:
        cluster1_arr, cluster2_arr = two_clusters_arrays(elements, thresholds, bin_half_size)
    else:  # cluster_number == 3
        cluster1_arr, cluster2_arr = three_clusters_arrays(elements, thresholds, bin_half_size)

    return cluster1_arr, cluster2_arr


@profile
def clustering(elements, bins_number=100):
    bin_half_size = (max(elements) - min(elements)) / (2 * bins_number)
    cluster1_arr, cluster2_arr = otsu_clustering(elements, 2, bins_number, bin_half_size)
    range_arr1 = max(cluster1_arr) - min(cluster1_arr)
    range_arr2 = max(cluster2_arr) - min(cluster2_arr)

    if RANGE_CHECK <= range_arr1 or RANGE_CHECK <= range_arr2:
        cluster1_arr, cluster2_arr = otsu_clustering(elements, 3, bins_number, bin_half_size)

    cluster1 = np.mean(cluster1_arr)
    cluster2 = np.mean(cluster2_arr)

    return cluster1, cluster2


@profile
def more_than_one(surrounding_mask):
    index = 0
    sm_len = len(surrounding_mask)
    more_ind = False
    previous_num = surrounding_mask[index]

    while not more_ind and index < sm_len:
        current_number = surrounding_mask[index]
        if current_number != previous_num:
            more_ind = True

        previous_num = current_number
        index += 1

    return more_ind


@profile
def threshold_front(r_img, mask_on_img, frontal_mask_all):
    img_x_dim, img_y_dim = r_img.shape[1], r_img.shape[0]
    mask_on_img_front = np.zeros((img_y_dim, img_x_dim))

    for x, y, z in zip(frontal_mask_all.x, frontal_mask_all.y, frontal_mask_all.z):
        surrounding_mask = neighbors_cells_z(mask_on_img, x, y, img_x_dim - 1, img_y_dim - 1)
        more_indication = more_than_one(surrounding_mask)
        mask_on_img_front[y, x] = 1

        if more_indication:
            cluster1, cluster2 = clustering(surrounding_mask)
            diff = cluster2 - cluster1
            threshold_buffer = diff * THRESHOLD_BUFFER
            threshold = cluster1 + threshold_buffer
            mask_on_img_front[y, x] = 0 if z < threshold else 1

    mask_marks = np.asarray(np.where(mask_on_img_front == 1)).T[:, [1, 0]]
    return mask_marks


@profile
def frontal_points(take_only_frontal_ind, r_img, h3d2i, df_with_bg, return_empty=False):
    if take_only_frontal_ind:
        arr = threshold_front(r_img, h3d2i, df_with_bg)
    else:
        if return_empty or df_with_bg is None:
            arr = []
        else:
            arr = df_with_bg[['x', 'y']].to_numpy()

    return arr


@profile
def get_frontal(r_img, df_3dh, h3d2i, mask_name, scale_factor):
    print('frontal: ', mask_name)
    frontal_main_mask_w_bg, frontal_add_mask_w_bg, frontal_rest_mask_w_bg = split_head_mask_parts(df_3dh, mask_name)

    # Check each point if it is came from frontal or hidden area of tha face
    famwb_arr = frontal_points(config[mask_name].mask_add_front_points_calc, r_img, h3d2i, frontal_add_mask_w_bg)
    fmmwb_arr = frontal_points(config[mask_name].mask_front_points_calc, r_img, h3d2i, frontal_main_mask_w_bg)
    frmwb_arr = frontal_points(config[mask_name].draw_rest_mask, r_img, h3d2i, frontal_rest_mask_w_bg, True)

    if mask_name == EYE_MASK_NAME:
        frontal_rest, frontal_add_mask, frontal_mask = scale(r_img, frmwb_arr, famwb_arr, fmmwb_arr, scale_factor)
    else:
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
    y = [left_filter_size, right_filter_dim]

    a, b = np.polyfit(x, y, 1)
    filter_dim = int(np.ceil(a * mask_size + b))
    if filter_dim < 1:
        filter_dim = 1
    filter_size = (filter_dim, filter_dim)

    return filter_size


def morphological_op(mask_x, mask_y, image, left_filter_size=config[EYE_MASK_NAME].filter_size,
                     right_filter_dim=FILTER_SIZE_MASK_RIGHT_POINT, morph_op=cv2.MORPH_CLOSE):
    mask_on_image = np.zeros((image.shape[0], image.shape[1]))
    for x, y in zip(mask_x, mask_y):
        mask_on_image[y, x] = 1

    # morphology close
    filter_size = calc_filter_size(mask_x, mask_y, left_filter_size, right_filter_dim)
    kernel = np.ones(filter_size, np.uint8)  # kernel filter
    morph_mask = cv2.morphologyEx(mask_on_image, morph_op, kernel)
    yy, xx = np.where(morph_mask == 1)

    return xx, yy


def add_forehead_mask(frontal_mask, frontal_rest, image):
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

    return rest_of_head_ind


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
    covid19_mask_ind, add_covid19_ind = make_covid19_mask(x, y, z)
    sunglasses_mask_ind, add_sunglasses_ind = make_sunglasses_mask(x, y)

    masks_order = [HAT_MASK_NAME, EYE_MASK_NAME, SCARF_MASK_NAME, COVID19_MASK_NAME, SUNGLASSES_MASK_NAME]
    masks_ind = [hat_mask_ind, eye_mask_ind, scarf_mask_ind, covid19_mask_ind, sunglasses_mask_ind]
    masks_add_ind = [None, None, None, add_covid19_ind, add_sunglasses_ind]

    rest_ind = [get_rest_mask(maskInd, maskAInd, vertices) for maskInd, maskAInd in zip(masks_ind, masks_add_ind)]
    masks_to_create = masks_name.split(',')
    head3d_cords = index_on_vertices(range(0, len(vertices)), vertices)
    add_mask_to_config(head3d_cords, masks_ind, masks_add_ind, rest_ind, masks_order, masks_to_create)

    return masks_to_create


def add_mask_to_config(head3d_cords, masks_ind, masks_add_ind, rest_ind, masks_order, masks_to_create):
    config[HEAD_3D_NAME] = head3d_cords
    for mask_name in masks_to_create:
        config[mask_name].mask_ind = masks_ind[masks_order.index(mask_name)]
        config[mask_name].mask_add_ind = masks_add_ind[masks_order.index(mask_name)]
        config[mask_name].rest_ind = rest_ind[masks_order.index(mask_name)]


def process_image(img_path, model, transform, masks_to_create, args):
    print(img_path)
    # Read an image
    img = cv2.imread(img_path, 1)

    # results from img2pose
    results = model.predict([transform(img)])[0]
    tic = time()

    # Get only one 6DOF from all the 6DFs that img2pose found
    pose, bbox = get_1id_pose(results, img, args.threshold)

    # face detected with img2pose and above the threshold
    if pose.size != 0:
        # Resize image that ROI will be in a fix size
        r_img, scale_factor = resize_image(img, bbox)

        # project 3D face according to pose
        df_3dh = project_3d(r_img, pose)

        # Projection of the 3d head z coordinate on the image
        h3d2i = head3d_z_dist(r_img, df_3dh)

        # for mask, mask_add, rest_of_head, mask_name in zip(masks, masks_add, rest_of_heads, MASKS_NAMES):
        for mask_name in masks_to_create:
            process_mask(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, img_path, bbox, args)
    else:
        print(f'No face detected for: {img_path}')
    config[HAT_MASK_NAME].mask_exists = False
    toc = time()
    return toc-tic

def process_mask(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, img_path, bbox, args):
    print('start: ',mask_name)

    # Get the location of the masks on the image
    mask_x, mask_y, rest_mask_x, rest_mask_y = render(img, r_img, df_3dh, h3d2i, mask_name, scale_factor)

    # The average color of the surrounding of the image
    color = bg_color(mask_x, mask_y, img)

    # Put the colored mask on the face in the image
    masked_image = color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name)

    # Save masked image
    save_image(img_path, mask_name, masked_image, args.output, bbox, args.bbox_ind, args.inc_bbox)
