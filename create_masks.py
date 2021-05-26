# todo: delete render_plot

import cv2
import numpy as np
from config import VERTICES_PATH, EYE_MASK_IND, HAT_MASK_IND, CORONA_MASK_IND, SCARF_MASK_IND, EYE_MASK, HAT_MASK
from project_on_image import transform_vertices


# This is the opposite of the function in expression-net-old
def get_hat_mask_index(a1, b1, c1, x_left, x_right, x, y):
    index_list = []
    for i in range(len(x)):
        if ((y[i] > (a1 * (x[i] ** 2) + b1 * x[i] + c1)) and
                (x[i] > x_left) and (x[i] < x_right)):  # or (x[i] <= x_left) or (x[i] >= x_right):

            index_list.append(i)

    return index_list


# This is the opposite of the function in expression-net-old
# get the scarf mask indexes on the model
def get_scarf_mask_index(a1, b1, c1, x_left, x_right, x, y):
    index_list = []
    for i in range(len(x)):
        if ((y[i] < (a1 * x[i] ** 2 + b1 * x[i] + c1)) and
                (x[i] > x_left) and (x[i] < x_right)):  # or (x[i] <= x_left) or (x[i] >= x_right):

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
    x_left = x[EYE_MASK_IND[0]]
    x_right = x[EYE_MASK_IND[1]]
    y_left = y[EYE_MASK_IND[0]]
    y_right = y[EYE_MASK_IND[1]]
    x_middle = x[EYE_MASK_IND[2]]
    y_chosen_top = y[EYE_MASK_IND[2]]
    y_chosen_down = y[EYE_MASK_IND[3]]

    x_3points = [x_left, x_middle, x_right]
    y_3points1 = [y_left, y_chosen_down, y_right]
    y_3points2 = [y_left, y_chosen_top, y_right]

    a1, b1, c1 = np.polyfit(x_3points, y_3points2, 2)
    a2, b2, c2 = np.polyfit(x_3points, y_3points1, 2)

    index_list = get_eyes_mask_index(a1, b1, c1, a2, b2, c2, x_left, x_right, x, y)

    return index_list


def make_scarf_mask(x, y):
    x_left = x[SCARF_MASK_IND[0]]
    x_right = x[SCARF_MASK_IND[1]]
    y_left = y[SCARF_MASK_IND[0]]
    y_right = y[SCARF_MASK_IND[1]]
    x_middle = x[SCARF_MASK_IND[2]]
    y_chosen_top = y[SCARF_MASK_IND[2]]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_chosen_top, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_scarf_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return index_list


# create hat mask
def make_hat_mask(x, y):
    x_left = x[HAT_MASK_IND[0]]
    x_right = x[HAT_MASK_IND[1]]
    y_left = y[HAT_MASK_IND[0]]
    y_right = y[HAT_MASK_IND[1]]
    x_middle = x[HAT_MASK_IND[2]]
    y_chosen_down = y[HAT_MASK_IND[2]]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_chosen_down, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_hat_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return index_list


def make_corona_mask(x, y, z):
    left_middle_ind = CORONA_MASK_IND[0]
    center_middle_ind = CORONA_MASK_IND[1]
    right_middle_ind = CORONA_MASK_IND[2]
    left_lower_ind = CORONA_MASK_IND[3]
    right_lower_ind = CORONA_MASK_IND[4]
    left_upper_string1 = CORONA_MASK_IND[5]
    left_upper_string2 = CORONA_MASK_IND[6]
    left_lower_string1 = CORONA_MASK_IND[7]
    left_lower_string2 = CORONA_MASK_IND[8]
    right_upper_string1 = CORONA_MASK_IND[9]
    right_upper_string2 = CORONA_MASK_IND[10]
    right_lower_string1 = CORONA_MASK_IND[11]
    right_lower_string2 = CORONA_MASK_IND[12]

    index_list1 = center_face_ind(left_middle_ind, center_middle_ind, right_middle_ind, \
                                  left_lower_ind, right_lower_ind, y, z)
    index_list2 = get_mask_string(left_upper_string1, left_upper_string2, 'LEFT', x, y, z)
    index_list3 = get_mask_string(left_lower_string1, left_lower_string2, 'LEFT', x, y, z)
    index_list4 = get_mask_string(right_upper_string1, right_upper_string2, 'RIGHT', x, y, z)
    index_list5 = get_mask_string(right_lower_string1, right_lower_string2, 'RIGHT', x, y, z)

    corona_mask_ind = index_list1
    corona_strings_ind = index_list2 + index_list3 + index_list4 + index_list5

    # face_indices = np.arange(len(x))
    # mask = np.ones_like(face_indices, bool)
    # mask[np.asarray(corona_mask_ind)] = False
    # visible_face_ind = face_indices[mask]

    return corona_mask_ind, corona_strings_ind


def center_face_ind(left_middle_ind, center_middle_ind, right_middle_ind, left_lower_ind, right_lower_ind, y, z):
    y_middle = y[[right_lower_ind, center_middle_ind, right_middle_ind]]
    z_middle = z[[right_lower_ind, center_middle_ind, right_middle_ind]]
    y_lower = y[[left_lower_ind, right_lower_ind]]
    z_lower = z[[left_lower_ind, right_lower_ind]]

    a, b, c = np.polyfit(y_middle, z_middle, 2)
    m, n = np.polyfit(y_lower, z_lower, 1)

    index_list = []
    for ii, y_i in enumerate(y):
        if ((z[ii] >= (a * (y_i ** 2) + b * y_i + c))) and (y[right_lower_ind] < y[ii] < y[right_middle_ind]) or \
                (z[ii] >= (m * y_i + n)) and (y[left_lower_ind] < y[ii] < y[right_lower_ind]):
            index_list.append(ii)

    return index_list


def get_mask_string(ind1, ind2, face_side, x, y, z):
    if face_side == 'LEFT':
        filtered_ind = [ii for ii, x_i in enumerate(x) if (x_i >= 0)]
    else:  # left face
        filtered_ind = [ii for ii, x_i in enumerate(x) if (x_i <= 0)]

    y_pos = y[filtered_ind]
    z_pos = z[filtered_ind]

    # Draw line
    received_y_pt = [y[ind1], y[ind2]]
    received_z_pt = [z[ind1], z[ind2]]
    m, mb = np.polyfit(received_y_pt, received_z_pt, 1)

    start_y = min(received_y_pt)
    end_y = max(received_y_pt)
    line_list = []
    string_size = 5
    for i in np.arange(start_y, end_y + 0.001, 0.001):
        distances = np.asarray(((z_pos - (m * i + mb)) ** 2 + (y_pos - i) ** 2))
        string_size_ind = np.argpartition(distances, string_size)[:string_size]
        cond_ind = list(np.asarray(filtered_ind)[string_size_ind])
        line_list += cond_ind

    return np.unique(np.asarray(line_list)).tolist()


def render(img, pose, mask_verts, mask_add_verts, rest_of_head_verts, mask_name):
    # Transform the 3DMM according to the pose
    mask_trans_vertices = transform_vertices(img, pose, mask_verts)
    rest_trans_vertices = transform_vertices(img, pose, rest_of_head_verts)

    # Whether to add the forehead to the mask, this is currenly only used for eye and hat masks
    if mask_name in [HAT_MASK, EYE_MASK]:
        mask_x, mask_y = add_headTop(img, mask_trans_vertices, rest_trans_vertices)
        mask_x, mask_y = np.append(mask_x.flatten(), mask_trans_vertices[:, 0]), \
                         np.append(mask_y.flatten(), mask_trans_vertices[:, 1])
        rest_x, rest_y = rest_trans_vertices[:, 0], rest_trans_vertices[:, 1]
    else:
        mask_x, mask_y = mask_trans_vertices[:, 0], mask_trans_vertices[:, 1]
        rest_x, rest_y = rest_trans_vertices[:, 0], rest_trans_vertices[:, 1]

    # Perform morphological close
    morph_mask_x, morph_mask_y = morphologicalClose(mask_x, mask_y, img)
    morph_rest_x, morph_rest_y = morphologicalClose(rest_x, rest_y, img)

    if mask_add_verts is not None:
        mask_add_trans_vertices = np.round(transform_vertices(img, pose, mask_add_verts)).astype(int)
        # Function that remove the
        morph_mask_x = np.append(morph_mask_x, mask_add_trans_vertices[:, 0])
        morph_mask_y = np.append(morph_mask_y, mask_add_trans_vertices[:, 1])

    return morph_mask_x, morph_mask_y, morph_rest_x, morph_rest_y


def render_plot(x, y, img, bboxes):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(x, y)
    for bbox in bboxes:
        plt.gca().add_patch(
            patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=3, edgecolor='b',
                              facecolor='none'))
    plt.show()


def index_on_vertices(index_list, vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    x_mask = x[index_list]
    y_mask = y[index_list]
    z_mask = z[index_list]
    size_array = (x_mask.shape[0], 3)
    maskSEP = np.zeros(size_array)
    maskSEP[:, 0] = x_mask
    maskSEP[:, 1] = y_mask
    maskSEP[:, 2] = z_mask

    return maskSEP


def bg_color(mask_x, mask_y, image):
    morph_mask_x, morph_mask_y = morphologicalClose(mask_x, mask_y, image)
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


def morphologicalClose(mask_x, mask_y, image):
    maskOnImage = np.zeros_like(image)
    for x, y in zip(mask_x, mask_y):
        if (0 <= x <= maskOnImage.shape[0] - 1) and (0 <= y <= maskOnImage.shape[1] - 1):
            maskOnImage[int(y), int(x), :] = [255, 255, 255]

    # morphology close
    gray_mask = cv2.cvtColor(maskOnImage, cv2.COLOR_BGR2GRAY)
    res, thresh_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)  # kernerl filter
    morph_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)
    yy, xx = np.where(morph_mask == 255)

    return xx, yy


def add_headTop(image, mask_trans_vertices, rest_trans_vertices):
    maskOnImage = np.zeros_like(image)
    mask_x_ind, mask_y_ind = mask_trans_vertices[:, 0].astype(int), mask_trans_vertices[:, 1].astype(int)
    for x, y in zip(mask_x_ind, mask_y_ind):
        maskOnImage[y, x] = 255

    bottom_hat = np.empty(image.shape[0])
    bottom_hat[:] = np.nan
    for i in range(image.shape[0]):
        iy, _ = np.where(maskOnImage[:, i])
        if iy.any():
            bottom_hat[i] = np.min(iy)

    print(mask_trans_vertices.shape, rest_trans_vertices.shape)
    all_face_proj = np.concatenate((mask_trans_vertices, rest_trans_vertices), axis=0)
    print(all_face_proj.shape)
    all_face_proj_y = all_face_proj[:, 1]
    max, min = np.max(all_face_proj_y), np.min(all_face_proj_y)
    kernel_len = int((max - min) / 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, kernel_len))
    dilated_image = cv2.dilate(maskOnImage, kernel, iterations=1)

    forehead_x_ind, forehead_y_ind = [], []
    for i in range(image.shape[0]):
        iy, _ = np.where(dilated_image[:, i])
        jj = np.where(iy <= bottom_hat[i])
        if jj[0] != []:
            j = iy[jj]
            forehead_x_ind.append([i] * len(j))
            forehead_y_ind.append(j)

    return np.asarray(forehead_x_ind), np.asarray(forehead_y_ind)


def get_rest_mask(maskInd, verts):
    rest_of_head_ind = np.setdiff1d(range(verts.shape[0]), maskInd)
    rest_of_head_mask = index_on_vertices(rest_of_head_ind, verts)

    return rest_of_head_mask


def load_3DMM():
    verts = np.load(VERTICES_PATH)
    th = np.pi
    R = [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
    verts_rotated = verts.copy()
    verts_rotated = np.matmul(verts_rotated, R)
    return verts, verts_rotated


def create_masks():
    # TODO: 1.not to use verts_rotated but verts
    #  todo: 2. chame Make**Mask funtions with out the None
    # todo: change the coorona creation code
    verts, verts_rotated = load_3DMM()
    x, y, z = verts_rotated[:, 0], verts_rotated[:, 1], verts_rotated[:, 2]
    eyeMaskInd = make_eye_mask(x, y)
    hatMaskInd = make_hat_mask(x, y)
    scarfMaskInd = make_scarf_mask(x, y)
    coronafMaskInd1, a = make_corona_mask(x, y, z)
    # coronafMaskInd2 = coronafMaskInd1.tolit()
    coronafMaskInd = coronafMaskInd1  # np.unique(np.asarray(coronafMaskInd1 + a)).tolist()
    # coronafMaskInd = np.setdiff1d(range(len(x)), coronafMaskInd3)   # THere is a hat on it

    masks_ind = [eyeMaskInd, hatMaskInd, scarfMaskInd, coronafMaskInd]
    masks_add_ind = [None, None, None, a]

    masks = [index_on_vertices(maskInd, verts) for maskInd in masks_ind]
    masks_additional = [None if mask_add_ind is None else \
                            index_on_vertices(mask_add_ind, verts) for mask_add_ind in masks_add_ind]
    rest_of_heads = [get_rest_mask(maskInd, verts) for maskInd in masks_ind]

    return masks, masks_additional, rest_of_heads
