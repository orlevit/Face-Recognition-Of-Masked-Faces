import os
import sys
import numpy as np
import cv2

sys.path.append('./img2pose')
from torchvision import transforms
from img2pose import img2poseModel
from model_loader import load_model
from config import DEPTH, MAX_SIZE, MIN_SIZE, POSE_MEAN, POSE_STDDEV, MODEL_PATH, PATH_3D_POINTS, DEST_PATH

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

