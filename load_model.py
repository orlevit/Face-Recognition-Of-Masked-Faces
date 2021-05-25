from config import DEPTH, MAX_SIZE, MIN_SIZE, POSE_MEAN, POSE_STDDEV, MODEL_PATH, PATH_3D_POINTS
import sys
sys.path.append('./img2pose')
import numpy as np
from torchvision import transforms
from img2pose import img2poseModel
from model_loader import load_model

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
