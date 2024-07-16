import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from masks_indices import make_eye_mask, make_hat_mask, make_covid19_mask, make_scarf_mask, make_sunglasses_mask
from helpers import split_head_mask_parts, get_1id_pose, resize_image, project_3d, color_face_mask, save_image, \
     head3d_z_dist, img_output_bbox, points_on_image, turn_to_odd, is_face_detected, max_connected_component, \
     scale_int_array, join_main_masks
from config_file import config, VERTICES_PATH, EYE_MASK_NAME, HAT_MASK_NAME, SCARF_MASK_NAME, COVID19_MASK_NAME, \
     SUNGLASSES_MASK_NAME, NEAR_NEIGHBOUR_STRIDE, MIN_MASK_SIZE, FILTER_MASK_RIGHT_POINT_IMAGE_SIZE, TOP_EYEMASK_INDS, \
     MASK_RIGHT_POINT, ADD_LEFT_POINT, ADD_RIGHT_POINT, MIN_POSE_OPEN_EYEMASK, RANGE_CHECK, MASK_EXTEND_BBOX_NORM, \
     HEAD_3D_NAME, THRESHOLD_BUFFER, MASK_EXTEND_PIXELS, EYE_HAT_MASK_LEFT_POINT, EYE_HAT_MASK_RIGHT_POINT, \
     ALL_SINGLE_MASKS

def render(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, bbox_ind, output_bbox, pose):
    if not config[mask_name].masks_combination:
        return render_mask(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, bbox_ind, output_bbox, pose)

    multi_morph_mask_x = np.array([], dtype=np.int64); multi_morph_mask_y = np.array([], dtype=np.int64)
    multi_morph_rest_x = np.array([]); multi_morph_rest_y = np.array([])
    for single_name in config[mask_name].masks_list:
        mask_x, mask_y, rest_mask_x, rest_mask_y = render_mask(img, r_img, df_3dh, h3d2i, single_name, scale_factor,
                                                               bbox_ind, output_bbox, pose)
        multi_morph_mask_x = np.append(multi_morph_mask_x, mask_x)
        multi_morph_mask_y = np.append(multi_morph_mask_y, mask_y)
        multi_morph_rest_x = np.append(multi_morph_rest_x, rest_mask_x)
        multi_morph_rest_y = np.append(multi_morph_rest_y, rest_mask_y)

    multi_morph_rest_x = multi_morph_rest_x[np.where(~np.isnan(multi_morph_rest_x.astype(float)))[0]]
    multi_morph_rest_y = multi_morph_rest_y[np.where(~np.isnan(multi_morph_rest_y.astype(float)))[0]]

    multi_morph_rest_x = multi_morph_rest_x.astype(int) if len(multi_morph_rest_x) else None
    multi_morph_rest_y = multi_morph_rest_y.astype(int) if len(multi_morph_rest_y) else None

    return multi_morph_mask_x, multi_morph_mask_y, multi_morph_rest_x, multi_morph_rest_y

def render_mask(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, bbox_ind, output_bbox, pose):
    # Get only frontal face areas
    frontal_mask, frontal_add_mask, frontal_rest = get_frontal(r_img, df_3dh, h3d2i, mask_name, scale_factor)

    # morphological operations on the mask
    mask_x, mask_y, mask_on_image = morphological_op(True, frontal_mask, img,
                                                     config[mask_name].mask_filter_size, MASK_RIGHT_POINT,
                                                     True, config[mask_name].main_mask_components_number)

    _, _, mask_add_on_image = morphological_op(config[mask_name].mask_add_front_points_calc,
                                                             frontal_add_mask, img, ADD_LEFT_POINT, ADD_RIGHT_POINT,
                                                             False, 0, cv2.MORPH_DILATE)

    morph_rest_x, morph_rest_y, rest_on_image = morphological_op(config[mask_name].draw_rest_mask, frontal_rest, img,
                                                     config[mask_name].rest_filter_size, MASK_RIGHT_POINT,
                                                     False, 0, cv2.MORPH_DILATE)

    if len(morph_rest_x) == 0:
        morph_rest_x, morph_rest_y = None, None

    # Whether to add the forehead to the mask, this is currently only used for eye and hat masks
    mask_on_image = forehead_mask(mask_name, frontal_mask, frontal_rest, bbox_ind, output_bbox, mask_on_image,
                                  mask_x, mask_y, img, df_3dh, scale_factor)

    # Extend the main part of the mask
    extended_morph_mask = extend_mask(mask_on_image, pose, output_bbox)
    if len(mask_add_on_image):
        extended_morph_mask = extended_morph_mask + mask_add_on_image

    # Extend the opening of the eye mask, when the is looking aside
    if mask_name == EYE_MASK_NAME and MIN_POSE_OPEN_EYEMASK < abs(pose[1]):
        extended_morph_rest = extend_mask(rest_on_image, pose, output_bbox)
        morph_rest_y, morph_rest_x = np.where(extended_morph_rest.astype(int))

        # mask without the extended eye opening
        extended_morph_mask = (extended_morph_mask + 2 * extended_morph_rest).astype(int)
        np.place(extended_morph_mask, extended_morph_mask != 1, 0)

    # Only one connecting component of the main mask
    morph_mask_one_component = max_connected_component(extended_morph_mask, True, 1)
    morph_mask_y, morph_mask_x = np.where(morph_mask_one_component.astype(int))

    return morph_mask_x, morph_mask_y, morph_rest_x, morph_rest_y


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

    return np.asarray(z_neighbors, dtype=np.float64)


def otsu_clustering(elements, bins_number, bin_half_size):
    threshold = threshold_multiotsu(elements, 2, nbins=bins_number)
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


def clustering(elements, bins_number=100):
    bin_half_size = (max(elements) - min(elements)) / (2 * bins_number)
    cluster1_arr, cluster2_arr = otsu_clustering(elements, bins_number, bin_half_size)
    range_arr1 = max(cluster1_arr) - min(cluster1_arr)
    range_arr2 = max(cluster2_arr) - min(cluster2_arr)

    if RANGE_CHECK <= range_arr2:
        cluster1_arr, cluster2_arr = otsu_clustering(cluster2_arr, bins_number, bin_half_size)
    elif RANGE_CHECK <= range_arr1 :
        _, cluster1_arr = otsu_clustering(cluster1_arr, bins_number, bin_half_size)

    cluster1 = np.mean(cluster1_arr)
    cluster2 = np.mean(cluster2_arr)

    return cluster1, cluster2


def more_than_one_points(surrounding_mask):
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


def threshold_front(r_img, mask_on_img, frontal_mask_all):
    img_x_dim, img_y_dim = r_img.shape[1], r_img.shape[0]
    mask_on_img_front = np.zeros((img_y_dim, img_x_dim))

    for x, y, z in zip(frontal_mask_all.x, frontal_mask_all.y, frontal_mask_all.z):
            mask_on_img_front[y, x] = 1
            surrounding_mask = neighbors_cells_z(mask_on_img, x, y, img_x_dim - 1, img_y_dim - 1)
            more_indication = more_than_one_points(surrounding_mask)

            if more_indication:
                cluster1, cluster2 = clustering(surrounding_mask)
                diff = cluster2 - cluster1
                threshold_buffer = diff * THRESHOLD_BUFFER
                threshold = cluster1 + threshold_buffer
                mask_on_img_front[y, x] = 0 if z < threshold else 1

    mask_marks = np.asarray(np.where(mask_on_img_front == 1)).T[:, [1, 0]]
    return mask_marks


def frontal_points(take_only_frontal_ind, r_img, h3d2i, df_with_bg):
    if take_only_frontal_ind and df_with_bg is not None:
        arr = threshold_front(r_img, h3d2i, df_with_bg)
    else:
        if df_with_bg is None:
            arr = []
        else:
            arr = df_with_bg[['x', 'y']].to_numpy()

    return arr


def get_frontal(r_img, df_3dh, h3d2i, mask_name, scale_factor):
    # print('frontal: ', mask_name)
    f_main_mask1_w_bg, f_main_mask2_w_bg ,f_add_mask_w_bg, f_rest_mask_w_bg = split_head_mask_parts(df_3dh, mask_name)

    # Check each point if it is came from frontal or hidden area of tha face
    famwb_arr = frontal_points(config[mask_name].mask_add_front_points_calc, r_img, h3d2i, f_add_mask_w_bg)

    fmmwb_arr1 = frontal_points(config[mask_name].mask_front_points_calc, r_img, h3d2i, f_main_mask1_w_bg)
    fmmwb_arr2 = frontal_points(config[mask_name].mask_front_points_calc, r_img, h3d2i, f_main_mask2_w_bg)
    fmmwb_arr = join_main_masks(fmmwb_arr1, fmmwb_arr2)

    frmwb_arr = frontal_points(config[mask_name].draw_rest_mask, r_img, h3d2i, f_rest_mask_w_bg)

    frontal_mask = scale_int_array(fmmwb_arr, scale_factor)
    frontal_add_mask = scale_int_array(famwb_arr, scale_factor)
    frontal_rest = scale_int_array(frmwb_arr, scale_factor)

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
    filter_dim = turn_to_odd(int(np.ceil(a * mask_size + b)))

    if filter_dim < 1:
        filter_dim = 1

    filter_size = (filter_dim, filter_dim)

    return filter_size


def extend_mask(mask_on_image, pose, output_bbox):
    w_bbox = output_bbox[2] - output_bbox[0]
    filter_size = max(int(round(MASK_EXTEND_PIXELS * abs(pose[1]) * w_bbox / MASK_EXTEND_BBOX_NORM)), 0)

    if pose[1] > 0:
        kernel = np.expand_dims(np.append(np.zeros(filter_size, np.uint8), np.ones(filter_size + 1, np.uint8)), axis=0)
    else:
        kernel = np.expand_dims(np.append(np.ones(filter_size + 1, np.uint8), np.zeros(filter_size, np.uint8)), axis=0)

    morph_mask = cv2.morphologyEx(mask_on_image.astype(np.uint8), cv2.MORPH_DILATE, kernel)

    return morph_mask


def morphological_op(morph_ind, mask, image, left_filter_size, right_filter_dim, c_ind, cn, morph_op=cv2.MORPH_CLOSE):
    if morph_ind:
        mask_x, mask_y = mask[:, 0], mask[:, 1]
        point_2d = points_on_image(mask_x, mask_y, image)
        filter_size = calc_filter_size(mask_x, mask_y, left_filter_size, right_filter_dim)
        kernel = np.ones(filter_size, np.uint8)
        morph_mask = cv2.morphologyEx(point_2d, morph_op, kernel)
        morph_mask_max = max_connected_component(morph_mask, c_ind, cn)
        yy, xx = np.where(morph_mask_max == 1)

    else:
        yy, xx, morph_mask_max = [], [], []

    return xx, yy, morph_mask_max


def add_forehead_mask(frontal_mask, frontal_rest, bbox_ind, output_bbox, mask_on_image):
    forehead_y, forehead_x = [], []
    frontal_mask_x, frontal_mask_y = frontal_mask[:, 0], frontal_mask[:, 1]
    min_proj_y = np.min(frontal_mask_y)

    if bbox_ind:
        kernel_len = max(2 * (min_proj_y - output_bbox[1]), 0)
    else:
        all_face_proj_y = np.concatenate((frontal_mask_y, frontal_rest[:, 1]), axis=0)
        max_proj_y =  np.max(all_face_proj_y)
        kernel_len = max((max_proj_y - min_proj_y) // 2, 0)

    if 1 < kernel_len:
        hkl = kernel_len // 2
        kernel = np.expand_dims(np.append(np.zeros(hkl, np.uint8), np.ones(hkl + 1, np.uint8)), axis=1)
        dilated_image = cv2.dilate(mask_on_image, kernel, iterations=1)
        full_hat_img = mask_on_image + dilated_image
        forehead_y, forehead_x= np.where(full_hat_img == 1)

    return forehead_x, forehead_y


def forehead_mask(mask_name, frontal_mask, frontal_rest, bbox_ind, output_bbox, mask_on_image,
                  mask_x, mask_y, img, df_3dh, scale_factor):

    if config[mask_name].add_forehead:
        if mask_name == HAT_MASK_NAME:
            f_x, f_y = add_forehead_mask(frontal_mask, frontal_rest, bbox_ind, output_bbox, mask_on_image)

            if not len(f_x):
                config[EYE_MASK_NAME].forehead_x, config[EYE_MASK_NAME].forehead_y = [], []
            else:
                # Save the relevant section of the forehead for eyemask
                df_top_eyemask = df_3dh.iloc[TOP_EYEMASK_INDS]
                top_eyemask = df_top_eyemask[~df_top_eyemask['x'].isnull()]
                top_eyemask_x = scale_int_array(top_eyemask.x.values, scale_factor)
                rel_top = np.where((np.min(top_eyemask_x) <= f_x) & (f_x <= np.max(top_eyemask_x)))
                config[EYE_MASK_NAME].forehead_x, config[EYE_MASK_NAME].forehead_y = f_x[rel_top], f_y[rel_top]
                mask_xwf, mask_ywf = np.append(f_x, mask_x).astype(int), np.append(f_y, mask_y).astype(int)
                mask_on_image = points_on_image(mask_xwf, mask_ywf, img)

        else: # Eyemask
            f_x, f_y = config[EYE_MASK_NAME].forehead_x, config[EYE_MASK_NAME].forehead_y
            mask_xwf, mask_ywf = np.append(f_x, mask_x).astype(int), np.append(f_y, mask_y).astype(int)
            mask_xywf = np.vstack((mask_xwf, mask_ywf)).T
            _, _, mask_on_image = morphological_op(True, mask_xywf, img, EYE_HAT_MASK_LEFT_POINT,
                                                   EYE_HAT_MASK_RIGHT_POINT, False, 0)

    return mask_on_image


def get_rest_mask(mask_ind1, mask_ind2, mask_add_ind, vertices_rotated, mask_name):
    if mask_ind2 is None:
        mask_ind = mask_ind1
    else:
        mask_ind = mask_ind1 + mask_ind2

    if mask_name == EYE_MASK_NAME:
        x, y = vertices_rotated[:, 0], vertices_rotated[:, 1]
        return make_eye_mask(x, y)

    relevant_indices = range(vertices_rotated.shape[0])
    rest_of_head_ind_with_add = np.setdiff1d(relevant_indices, mask_ind)
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

    hat_mask_ind = make_hat_mask(x, y)
    scarf_mask_ind = make_scarf_mask(x, y)
    covid19_mask_ind, add_covid19_ind = make_covid19_mask(x, y, z)
    eye_mask_ind = [ii for ii, cord in enumerate(z) if (cord >= 0)]
    sunglasses_lenses_left_ind, sunglasses_lenses_right_ind, add_sunglasses_ind = make_sunglasses_mask(x, y)

    masks_order = [HAT_MASK_NAME, EYE_MASK_NAME, SCARF_MASK_NAME, COVID19_MASK_NAME, SUNGLASSES_MASK_NAME]
    masks_ind1 = [hat_mask_ind, eye_mask_ind, scarf_mask_ind, covid19_mask_ind, sunglasses_lenses_left_ind]
    masks_ind2 = [None, None, None, None, sunglasses_lenses_right_ind]
    masks_add_ind = [None, None, None, add_covid19_ind, add_sunglasses_ind]
    rest_ind = [get_rest_mask(maskInd1, maskInd2, maskAInd, vertices_rotated, mask_name)
                for maskInd1, maskInd2, maskAInd, mask_name in zip(masks_ind1, masks_ind2, masks_add_ind, masks_order)]
    masks_to_create = masks_name.split(',')
    masks_to_store = ALL_SINGLE_MASKS.split(',')
    head3d_cords = index_on_vertices(range(0, len(vertices)), vertices)
    add_mask_to_config(head3d_cords, masks_ind1, masks_ind2, masks_add_ind, rest_ind, masks_order, masks_to_store)

    return masks_to_create


def add_mask_to_config(head3d_cords, masks_ind1, masks_ind2, masks_add_ind, rest_ind, masks_order, masks_to_store):
    config[HEAD_3D_NAME] = head3d_cords
    for mask_name in masks_to_store:
        config[mask_name].mask_ind1 = masks_ind1[masks_order.index(mask_name)]
        config[mask_name].mask_ind2 = masks_ind2[masks_order.index(mask_name)]
        config[mask_name].mask_add_ind = masks_add_ind[masks_order.index(mask_name)]
        config[mask_name].rest_ind = rest_ind[masks_order.index(mask_name)]


def process_image(img_path, model, transform, masks_to_create, args):
    # Read an image
    img = cv2.imread(img_path, 1)

    # results from img2pose
    results = model.predict([transform(img)])[0]

    # Get only one 6DOF from all the 6DFs that img2pose found
    pose, bbox = get_1id_pose(results, img, args.threshold)

    # Indication whether a face was detected the successfully
    face_detected_indication = is_face_detected(img, bbox)

    # face detected with img2pose and above the threshold
    if face_detected_indication:
        # Resize image that ROI will be in a fix size
        r_img, scale_factor = resize_image(img, bbox)

        # output image selected area
        output_bbox = img_output_bbox(img, bbox, args.inc_bbox, args.bbox_ind)

        # project 3D face according to pose
        df_3dh = project_3d(r_img, pose)

        # Projection of the 3d head z coordinate on the image
        h3d2i = head3d_z_dist(r_img, df_3dh)

        # for mask, mask_add, rest_of_head, mask_name in zip(masks, masks_add, rest_of_heads, MASKS_NAMES):
        for mask_name in masks_to_create:
            process_mask(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, img_path, args, output_bbox, pose)
    else:
        print(f'No face detected for: {img_path}')


def process_mask(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, img_path, args, output_bbox, pose):
    # Get the location of the masks on the image
    mask_x, mask_y, rest_mask_x, rest_mask_y = \
        render(img, r_img, df_3dh, h3d2i, mask_name, scale_factor, args.bbox_ind, output_bbox, pose)

    # The average color of the surrounding of the image
    color = bg_color(mask_x, mask_y, img)

    # Put the colored mask on the face in the image
    masked_image = color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name)

    # Save masked image
    save_image(img_path, mask_name, masked_image, args.output, args.bbox_ind, output_bbox)
