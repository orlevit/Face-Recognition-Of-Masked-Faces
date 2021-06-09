import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append('./img2pose')
from torchvision import transforms
from img2pose import img2poseModel
from model_loader import load_model
from config_file import DEPTH, MAX_SIZE, MIN_SIZE, POSE_MEAN, POSE_STDDEV, MODEL_PATH, \
    PATH_3D_POINTS, ALL_MASKS


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


def save_image(img_path, mask_name, img_output, output, img_input, bbox):
    # Extracts the right directory to create in the destination
    full_path, image_name = os.path.split(os.path.normpath(img_path))
    image_org_dir = os.path.basename(full_path)
    image_dst_dir = os.path.join(output, mask_name, image_org_dir)
    image_dst = os.path.join(image_dst_dir, image_name)

    image_dst_dir2 =output# os.path.join(output, image_org_dir)
    image_dst2 = os.path.join(image_dst_dir2, image_name)
    if not os.path.exists(image_dst_dir2):
        os.makedirs(image_dst_dir2)
    # Create the directory if it doesn't exists
    if not os.path.exists(image_dst_dir):
        os.makedirs(image_dst_dir)

    # Save the image
    img = draw_bbox(img_input, bbox)
    # cv2.imwrite(image_dst, img_output)
    cv2.imwrite(image_dst2, img)

def draw_bbox(img, bbox):
    def  inc_bbox(bbox, inc, img, color):
        wbbox = bbox[2] - bbox[0]
        lbbox = bbox[3] - bbox[1]
        half_w = wbbox // 2
        half_l = lbbox // 2
        half_w_inc = half_w * (1 + inc)
        half_l_inc = half_l * (1 + inc)
        cx = half_w + bbox[0]
        cy = half_l + bbox[1]
        n0 = max(np.round(cx - half_w_inc).astype(int), 0)
        n1 = max(np.round(cy - half_l_inc).astype(int), 0)
        n2 = min(np.round(cx + half_w_inc).astype(int), img.shape[0] - 1)
        n3 = min(np.round(cy + half_l_inc).astype(int), img.shape[1] - 1)

        for x in range(n0, n2):
            img[n1, x, :] = color
            img[n3, x, :] = color

        for y in range(n1, n3):
            img[y, n0, :] = color
            img[y, n2, :] = color

        return img

    img = inc_bbox(bbox, 0.25, img, [0, 0, 255])
    img = inc_bbox(bbox, 0.5, img, [225, 0, 0])

    for x in range(int(bbox[0]), int(bbox[2])):
        img[max(int(bbox[1]),0), x, :] = [0, 255, 0]
        img[min(int(bbox[3]),img.shape[1] - 1), x, :] = [0, 255, 0]


    for y in range(int(bbox[1]), int(bbox[3])):
        img[y, max(int(bbox[0]), 0), :] = [0, 255, 0]
        img[y, min(int(bbox[2]),img.shape[0] - 1), :] = [0, 255, 0]


    return img
    # Create figure and axes
    # fig, ax = plt.subplots()
    #
    # # Display the image
    # ax.imshow(img)
    #
    # # Create a Rectangle patch
    # rect1 = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r',
    #                           facecolor='none')
    # rect2 = patches.Rectangle((n0, n1), n2 - n0, n3 - n1, linewidth=1, edgecolor='g', facecolor='none')
    # ax.scatter(bbox[2], bbox[3])
    # # Add the patch to the Axes
    # ax.add_patch(rect1)
    # ax.add_patch(rect2)
    #
    # plt.show()


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
        # if more than one identity recognized, then check who has the biggest condition:
        # bounding box area - (distance of bounding box center from image center)^4
        bbox = results["boxes"].cpu().numpy()[possible_id_ind].astype('float')
        bounding_box_size = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        offsets = np.vstack([(bbox[:, 0] + bbox[:, 2]) / 2 - img_w_center, (bbox[:, 1] + bbox[:, 3]) / 2 - img_h_center])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        bbox_idx = np.argmax(bounding_box_size - offset_dist_squared * 2.0)

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


def mask_on_img(mask_x, mask_y, img, color):
    for x, y in zip(mask_x, mask_y):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x, :] = [color[0], color[1], color[2]]

    return img


def rest_on_img(rest_mask_x, rest_mask_y, img, img_output):
    for x, y in zip(rest_mask_x, rest_mask_y):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img_output[y, x, :] = img[y, x, :]

    return img_output


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory with input images or csv file with paths.')
    parser.add_argument('-o', '--output', type=str, help='Output directory.')
    parser.add_argument('-e', '--image_extensions', default='.jpg,.bmp,.jpeg,.png',
                        type=str, help='The extensions of the images.')
    parser.add_argument('-m', '--masks', default=ALL_MASKS, type=str, help='Which masks to create.')
    parser.add_argument('-t', '--threshold', default=0.0, type=float,
                        help='The minimum confidence score for img2pose for face detection')

    return parser.parse_args()

