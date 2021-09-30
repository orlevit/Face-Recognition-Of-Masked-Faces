import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob

sys.path.append('./img2pose')
from torchvision import transforms
from img2pose import img2poseModel
from model_loader import load_model
from project_on_image import transform_vertices
from config_file import config, DEPTH, MAX_SIZE, MIN_SIZE, POSE_MEAN, POSE_STDDEV, MODEL_PATH, \
    PATH_3D_POINTS, ALL_MASKS, BBOX_REQUESTED_SIZE, EYE_MASK_NAME
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


def save_image(img_path, mask_name, img_output, output, bbox, bbox_ind, inc_bbox):
    # Extracts the right directory to create in the destination
    full_path, image_name = os.path.split(os.path.normpath(img_path))
    image_org_dir = os.path.basename(full_path)
    image_dst_dir = os.path.join(output, mask_name, image_org_dir)
    image_dst = os.path.join(image_dst_dir, image_name)

    # Create the directory if it doesn't exists
    if not os.path.exists(image_dst_dir):
        os.makedirs(image_dst_dir)

    # Save the image
    if bbox_ind:
        img_output = crop_bbox(img_output, bbox, inc_bbox)

    cv2.imwrite(image_dst, img_output)


def resize_image(image, bbox):
    w_bbox = bbox[2] - bbox[0]
    h_bbox = bbox[3] - bbox[1]
    max_dim = max(w_bbox, h_bbox)
    scale_img = BBOX_REQUESTED_SIZE / max_dim
    h_scaled = int(image.shape[0] * scale_img)
    w_scaled = int(image.shape[1] * scale_img)
    resized_image = cv2.resize(image, (w_scaled, h_scaled))

    return resized_image, scale_img


def masks_parts_dataframe(r_img, pose, mask_name):
    # An indication whether it is a mask coordinate, additional  mask or rest of the head and add them to the matrices
    mask_marks = 3 * np.ones([config[mask_name].mask_ind.shape[0], 1], dtype=bool)
    mask_stacked = np.hstack((config[mask_name].mask_ind, mask_marks))
    rest_marks = np.ones([config[mask_name].rest_ind.shape[0], 1], dtype=bool)
    rest_stacked = np.hstack((config[mask_name].rest_ind, rest_marks))

    if isinstance(config[mask_name].mask_add_ind, type(None)):
        combined_float = np.vstack((mask_stacked, rest_stacked))
    else:
        mask_add_marks = 2 * np.ones([config[mask_name].mask_add_ind.shape[0], 1], dtype=bool)
        mask_add_stacked = np.hstack((config[mask_name].mask_add_ind, mask_add_marks))
        combined_float = np.vstack((mask_stacked, mask_add_stacked, rest_stacked))

    # Masks projection on the image plane
    combined_float[:, :3] = transform_vertices(r_img, pose, combined_float[:, :3])

    # turn values from float to integer
    combined = np.round(combined_float).astype(int)
    df = pd.DataFrame(combined, columns=['x', 'y', 'z', 'mask'])
    df_in_range = df[((0 <= df.x) & (df.x <= r_img.shape[1] - 1)) & ((0 <= df.y) & (df.y <= r_img.shape[0] - 1))]

    return df_in_range


@profile
def scale(img, most_important_mask, second_important_mask, third_important_mask, scale_factor):
    most_mask_img = mark_image_with_mask(most_important_mask, img, scale_factor)
    second_mask_img = mark_image_with_mask(second_important_mask, img, scale_factor)
    third_img = mark_image_with_mask(third_important_mask, img, scale_factor)

    mask_on_image = np.multiply(4, most_mask_img) + \
                    np.multiply(2, second_mask_img) +\
                    np.multiply(1, third_img)

    # Each pixel is main mask/additional strings or rest of the head, the numbers 1/2/4 are arbitrary,
    # and used to get the relevant type even when there are overlapping
    most_mask = np.asarray(np.where(np.isin(mask_on_image, [4, 5, 6, 7])))[[1, 0], :].T
    second_mask = np.asarray(np.where(np.isin(mask_on_image, [2, 3])))[[1, 0], :].T
    third_rest = np.asarray(np.where(mask_on_image == 1))[[1, 0], :].T

    return most_mask, second_mask, third_rest


def crop_bbox(img, bbox, inc_bbox):
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
    n2 = min(np.round(cx + half_w_inc).astype(int), img.shape[1] - 1)
    n3 = min(np.round(cy + half_l_inc).astype(int), img.shape[0] - 1)

    return img[n1:n3, n0:n2, :]


def get_1id_pose(results, img, threshold):
    h, w, _ = img.shape
    img_h_center = h / 2
    img_w_center = w / 2

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
        bboxes = results["boxes"].cpu().numpy()[possible_id_ind].astype('float')
        scores = results["scores"].cpu().numpy()[possible_id_ind].astype('float')
        bounding_box_size = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        center_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        dist_x = np.min(np.vstack((center_x, abs(center_x - img.shape[1]))), axis=0)
        dist_y = np.min(np.vstack((center_y, abs(center_y - img.shape[0]))), axis=0)
        offsets = np.vstack([dist_x, dist_y])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        bbox_idx = np.argmax(scores * (bounding_box_size + offset_dist_squared))

        return all_dofs[bbox_idx], all_bboxes[bbox_idx]


def read_images(input_images, image_extensions):
    if os.path.isfile(input_images):
        img_paths = pd.read_csv(input_images, delimiter=" ", header=None)
        img_paths = np.asarray(img_paths).squeeze()
    else:
        img_paths = [image for ext in image_extensions.split(',')
                     for image in glob(os.path.join(input_images, '**', '*' + ext), recursive=True)]

    return img_paths


def color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name, config):
    img_output = mask_on_img(mask_x, mask_y, img.copy(), color)
    if config[mask_name].draw_rest_mask:
        img_output = rest_on_img(rest_mask_x, rest_mask_y, img, img_output)

    return img_output


def mark_image_with_mask(frontal_coords, img, scale_factor):
    mask_on_image = [0]

    if len(frontal_coords):
        frontal_coords_scaled = (frontal_coords / scale_factor).astype(int)
        img_y_dim, img_x_dim = int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor)
        mask_on_image = np.zeros((img_y_dim, img_x_dim))
        for x, y in zip(frontal_coords_scaled[:, 0], frontal_coords_scaled[:, 1]):
            mask_on_image[min(y, img_y_dim - 1), min(x, img_x_dim - 1)] = 1

    return mask_on_image


def mask_on_img(mask_x, mask_y, img, color):
    for x, y in zip(mask_x, mask_y):
        img[y, x, :] = [color[0], color[1], color[2]]

    return img


def rest_on_img(rest_mask_x, rest_mask_y, img, img_output):
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
