import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from scipy.spatial.transform import Rotation

from time import time


sys.path.append('./img2pose')
from torchvision import transforms
from img2pose import img2poseModel
from model_loader import load_model
from project_on_image import transform_vertices
from config_file import config, DEPTH, MAX_SIZE, MIN_SIZE, POSE_MEAN, POSE_STDDEV, MODEL_PATH, PATH_3D_POINTS, \
    ALL_MASKS, BBOX_REQUESTED_SIZE, HEAD_3D_NAME, SLOPE_TRAPEZOID, INTERCEPT_TRAPEZOID, MIN_TRAPEZOID_INPUT, \
    MIN_TRAPEZOID_OUTPUT, YAW_IMPORTANCE, PITCH_IMPORTANCE, MIN_POSE_SCORES, MIN_MASK_PIXELS, \
    MIN_DETECTED_FACE_PERCENTAGE

from line_profiler_pycharm import profile


def get_model():
    transform = transforms.Compose([transforms.ToTensor()])
    threed_points = np.load(PATH_3D_POINTS)
    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)

    img2pose_model = img2poseModel(
        DEPTH, MIN_SIZE, MAX_SIZE,
        pose_mean=pose_mean, pose_stddev=pose_stddev,
        threed_68_points=threed_points,
    )
    load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
    img2pose_model.evaluate()

    return img2pose_model, transform


def save_image(img_path, mask_name, img_output, output, output_bbox):
    # Extracts the right directory to create in the destination
    full_path, image_name = os.path.split(os.path.normpath(img_path))
    image_org_dir = os.path.basename(full_path)
    image_dst_dir = os.path.join(output, mask_name, image_org_dir)
    image_dst = os.path.join(image_dst_dir, image_name)

    # Create the directory if it doesn't exists
    if not os.path.exists(image_dst_dir):
        os.makedirs(image_dst_dir)

    # Extract from the image the wanted area
    img_output_bbox = img_output[output_bbox[1]: output_bbox[3], output_bbox[0]:output_bbox[2], :]

    # Save the image
    cv2.imwrite(image_dst, img_output_bbox)


def resize_image(image, bbox):
    w_bbox = bbox[2] - bbox[0]
    h_bbox = bbox[3] - bbox[1]
    max_dim = max(w_bbox, h_bbox)
    scale_img = BBOX_REQUESTED_SIZE / max_dim
    h_scaled = int(image.shape[0] * scale_img)
    w_scaled = int(image.shape[1] * scale_img)
    resized_image = cv2.resize(image, (w_scaled, h_scaled))

    return resized_image, scale_img


@profile
def split_head_mask_parts(df_3dh, mask_name):
    frontal_main_mask_with_bg = head3d_to_mask(df_3dh, mask_name, "mask_ind")

    if config[mask_name].mask_add_ind is not None:
        frontal_add_mask_with_bg = head3d_to_mask(df_3dh, mask_name, "mask_add_ind")
    else:
        frontal_add_mask_with_bg = None

    if config[mask_name].draw_rest_mask is not None:
        frontal_rest_mask_with_bg = head3d_to_mask(df_3dh, mask_name, "rest_ind")
    else:
        frontal_rest_mask_with_bg = None

    return frontal_main_mask_with_bg, frontal_add_mask_with_bg, frontal_rest_mask_with_bg


def max_continuous_area(morph_mask, make_contour_ind, contours_number):
    if not make_contour_ind:
        return morph_mask

    morph_mask_max = np.zeros_like(morph_mask)
    contours, _ = cv2.findContours(morph_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    relevant_contour_indices = [index for index, contour in enumerate(contours) if MIN_MASK_PIXELS < len(contour)]
    relevant_contours_number = min(len(relevant_contour_indices), contours_number)

    contour_lengths = list(map(len, contours))
    contours_indices = np.argpartition(contour_lengths, -relevant_contours_number)[-relevant_contours_number:]
    relevant_contours = [contours[index] for index in contours_indices]
    cv2.drawContours(morph_mask_max, relevant_contours, -1, color=1, thickness=-1)

    return morph_mask_max


def turn_to_odd(num):
    if num & 1:
        return num

    return num + 1


@profile
def head3d_to_mask(df_3dh, mask_name, mask_ind):
    df_mask_with_nulls = df_3dh.iloc[config[mask_name][mask_ind]]
    df_mask = df_mask_with_nulls[~df_mask_with_nulls['x'].isnull()].astype(int)
    mask_with_bg = df_mask.sort_values(['z'], ascending=False).drop_duplicates(['x', 'y'], keep='first')

    return mask_with_bg


@profile
def project_3d(r_img, pose):
    # Masks projection on the image plane
    projected_head_float = transform_vertices(r_img, pose, config[HEAD_3D_NAME])

    ####################3333
    # pose2 = pose.copy()
    # pose2[-1] *= 0.98
    # projected_head_float2 = transform_vertices(r_img, pose2, config[HEAD_3D_NAME][config["covid19mask"].mask_add_ind])
    # projected_head_float[config["covid19mask"].mask_add_ind, :] = projected_head_float2
    #####################33


    # turn values from float to integer
    projected_head = np.round(projected_head_float).astype(int)

    df = pd.DataFrame(projected_head, columns=['x', 'y', 'z'])
    values_in_range = ((0 <= df.x) & (df.x <= r_img.shape[1] - 1)) & ((0 <= df.y) & (df.y <= r_img.shape[0] - 1))
    df[~values_in_range] = [None, None, None]

    return df


def img_output_bbox(img, bbox, inc_bbox, bbox_ind):
    img_x_dim = img.shape[1]
    img_y_dim = img.shape[0]

    if not bbox_ind:
        return [0, 0, img_x_dim, img_y_dim]

    wbbox = bbox[2] - bbox[0]
    lbbox = bbox[3] - bbox[1]
    half_w = wbbox / 2
    half_l = lbbox / 2
    half_w_inc = half_w * (1 + inc_bbox)
    half_l_inc = half_l * (1 + inc_bbox)
    cx = half_w + bbox[0]
    cy = half_l + bbox[1]
    n0 = max(np.round(cx - half_w_inc).astype(int), 0)
    n1 = max(np.round(cy - half_l_inc).astype(int), 0)
    n2 = min(np.round(cx + half_w_inc).astype(int), img_x_dim - 1)
    n3 = min(np.round(cy + half_l_inc).astype(int), img_y_dim - 1)

    return [n0, n1, n2, n3]


# return pitch, yaw, roll
def rotvec_to_euler(poses):
    poses_rotvec = poses[:, :3]
    rotvec = Rotation.from_rotvec(poses_rotvec).as_matrix()
    rotvec_transposed = np.array(list(map(lambda x: np.transpose(x), rotvec)))
    angles = Rotation.from_matrix(rotvec_transposed).as_euler('xyz', degrees=True)
    angles[0, 1:3] *= -1

    return angles


def trapezoid(x):
    if MIN_TRAPEZOID_INPUT < abs(x):
        return max(SLOPE_TRAPEZOID * abs(x) + INTERCEPT_TRAPEZOID, MIN_TRAPEZOID_OUTPUT)
    return 1


def pose_scores(poses):
    angles = rotvec_to_euler(poses)
    pitches, yaws = angles[:, 0], angles[:, 1]
    vtrapezoid = np.vectorize(trapezoid)
    composition = YAW_IMPORTANCE * vtrapezoid(yaws) + PITCH_IMPORTANCE * vtrapezoid(pitches)
    scores = np.where(MIN_POSE_SCORES <= composition, composition, MIN_POSE_SCORES)

    return scores


def get_1id_pose(results, img, threshold):
    h, w, _ = img.shape

    # Get all the bounding boxes
    all_bboxes = results["boxes"].cpu().numpy().astype('float')
    all_dofs = results["dofs"].cpu().numpy().astype('float')

    # only the bounding boxes that have sufficient threshold
    possible_id_ind = [i for i in range(len(all_bboxes)) if results["scores"][i] > threshold]

    # If only one identity recognized, return it
    if len(possible_id_ind) == 1:
        return all_dofs[possible_id_ind[0]], all_bboxes[possible_id_ind[0]]

    # If zero identities were recognized
    elif not possible_id_ind:
        return np.array([]), np.array([])

    else:
        # if more than one identity recognized, then check who has the biggest condition
        poses = results["dofs"].cpu().numpy()[possible_id_ind].astype('float')
        bboxes = results["boxes"].cpu().numpy()[possible_id_ind].astype('float')
        i_scores = results["scores"].cpu().numpy()[possible_id_ind].astype('float')
        bounding_box_size = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        center_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        dist_x = np.min(np.vstack((center_x, abs(center_x - img.shape[1]))), axis=0)
        dist_y = np.min(np.vstack((center_y, abs(center_y - img.shape[0]))), axis=0)
        offsets = np.vstack([dist_x, dist_y])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        p_scores = pose_scores(poses)
        bbox_idx = np.argmax(i_scores * p_scores * (bounding_box_size + offset_dist_squared))

        return all_dofs[bbox_idx], all_bboxes[bbox_idx]


def is_face_detected(img, bbox):
    image_h, image_w, _ = img.shape

    h_bbox = bbox[3] - bbox[1]
    w_bbox = bbox[2] - bbox[0]

    image_size = image_h * image_w
    bbox_size = h_bbox * w_bbox

    area_percentage = bbox_size / image_size
    print('\n',np.round(area_percentage,4),w_bbox,h_bbox,np.round(bbox_size,4),image_size,'\n')
    return MIN_DETECTED_FACE_PERCENTAGE < area_percentage


def read_images(input_images, image_extensions):
    if os.path.isfile(input_images):
        img_paths = pd.read_csv(input_images, delimiter=" ", header=None)
        img_paths = np.asarray(img_paths).squeeze()
    else:
        img_paths = [image for ext in image_extensions.split(',')
                     for image in glob(os.path.join(input_images, '**', '*' + ext), recursive=True)]

    return img_paths


def color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name):
    img_output = draw_mask_on_img(mask_x, mask_y, img.copy(), color)
    if config[mask_name].draw_rest_mask:
        img_output = draw_rest_on_img(rest_mask_x, rest_mask_y, img, img_output)

    return img_output


def points_on_image(points_x, points_y, image):
    mask_on_image = np.zeros((image.shape[0], image.shape[1]))
    for x, y in zip(points_x, points_y):
        mask_on_image[y, x] = 1

    return mask_on_image


@profile
def head3d_z_dist(r_img, df):
    img_x_dim, img_y_dim = r_img.shape[1], r_img.shape[0]
    mask_on_img = np.asarray([[None] * img_x_dim] * img_y_dim)
    not_none_df = df[~df['x'].isnull()].astype(int)

    # Each pixel contains a list of all the Z coordinates from the 3D model
    for x, y, z in zip(not_none_df.x, not_none_df.y, not_none_df.z):
        if mask_on_img[y, x] is None:
            mask_on_img[y, x] = [z]
        else:
            mask_on_img[y, x].append(z)

    return mask_on_img


def draw_mask_on_img(mask_x, mask_y, img, color):
    for x, y in zip(mask_x, mask_y):
        img[y, x, :] = [color[0], color[1], color[2]]

    return img


def draw_rest_on_img(rest_mask_x, rest_mask_y, img, img_output):
    for x, y in zip(rest_mask_x, rest_mask_y):
        img_output[y, x, :] = img[y, x, :]

    return img_output


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory with input images or csv file with paths.')
    parser.add_argument('-o', '--output', type=str, help='Output directory.')
    parser.add_argument('-e', '--image-extensions', default='.jpg,.bmp,.jpeg,.png',
                        type=str, help='The extensions of the images.')
    parser.add_argument('-m', '--masks', default=ALL_MASKS, type=str, help='Which masks to create.')
    parser.add_argument('-t', '--threshold', default=0.0, type=float,
                        help='The minimum confidence score for img2pose for face detection')
    parser.add_argument('-b', '--bbox-ind', default=True, type=str2bool, help='Return the original or cropped'
                                                                              'bounding box image with mask')
    parser.add_argument('-inc', '--inc-bbox', default=0.25, type=float, help='The increase of the bbox in percent')
    parser.add_argument('-ch', '--chunk-size', default=100, type=int, help='The chunk size per worker')
    parser.add_argument('-cpu', '--cpu-num', default=os.cpu_count(), type=int, help='Number of CPUs in multiprocessing')

    return parser.parse_args()
