import os
import sys
import numpy as np
import cv2

sys.path.append('./img2pose')
from torchvision import transforms
from img2pose import img2poseModel
from model_loader import load_model
from config import DEPTH, MAX_SIZE, MIN_SIZE, POSE_MEAN, POSE_STDDEV, MODEL_PATH, PATH_3D_POINTS,\
				   DEST_PATH, THRESHOLD, IMAGES_PATH, CORONA_MASK

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

def save_image(img_path, mask_name, img_output):

	# Extracts the right directory to create in the destination
	full_path, image_name = os.path.split(os.path.normpath(img_path))
	image_org_dir = os.path.basename(full_path)
	image_dst_dir = DEST_PATH + '//' + mask_name + '//' + image_org_dir
	image_dst = image_dst_dir + '//' + image_name

	# Create the directory if it doesn't exists
	if not os.path.exists(image_dst_dir):
		os.makedirs(image_dst_dir)

	# Save the image
	cv2.imwrite(image_dst, img_output)

def get_1id_pose(results, img):
	h, w, _ = img.shape
	img_h_center = h / 2
	img_w_center = w / 2

	# Get all the bounding boxes
	all_bboxes = results["boxes"].cpu().numpy().astype('float')

	# only the bounding boxes that have sufficient threshold
	possible_id_ind = [i for i in range(len(all_bboxes)) if results["scores"][i] > THRESHOLD]

	# If only one identity recognized, return it
	if len(possible_id_ind) == 1:
		return results["dofs"].cpu().numpy()[possible_id_ind].squeeze().astype('float')

	# if more than one identity recognized, then check who has the biggest condition:
	# bounding box area - (distance of bounding box center from image center)^4
	bbox = results["boxes"].cpu().numpy()[possible_id_ind].astype('float')
	bounding_box_size = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
	offsets = np.vstack([(bbox[:, 0] + bbox[:, 2]) / 2 - img_w_center, (bbox[:, 1] + bbox[:, 3]) / 2 - img_h_center])
	offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
	bbox_idx = np.argmax(bounding_box_size - offset_dist_squared * 2.0)

	return results["dofs"].cpu().numpy()[bbox_idx].squeeze().astype('float')

def read_images():
	if os.path.isfile(IMAGES_PATH):
		img_paths = pd.read_csv(IMAGES_PATH, delimiter=" ", header=None)
		img_paths = np.asarray(img_paths).squeeze()
	else:
		img_paths = [os.path.join(IMAGES_PATH, img_path) for img_path in os.listdir(IMAGES_PATH)]

	return img_paths

def color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name):
	img_output = img.copy()
	# todo: change the int(x) anf int(y) to x and y - also to the round
	if mask_name == CORONA_MASK:
		for x, y in zip(rest_mask_x, rest_mask_y):
			img_output[round(y), round(x), :] = img[round(y), round(x), :]
		for x, y in zip(mask_x, mask_y):
			img_output[round(y), round(x), :] = [color[0], color[1], color[2]]  # BGR
	else:
		for x, y in zip(mask_x, mask_y):
			img_output[round(y), round(x), :] = [color[0], color[1], color[2]]  # BGR
		for x, y in zip(rest_mask_x, rest_mask_y):
			img_output[round(y), round(x), :] = img[round(y), round(x), :]

	return img_output