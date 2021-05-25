from config import DEPTH, MAX_SIZE, MIN_SIZE, POSE_MEAN, POSE_STDDEV

def load_model():
	transform = transforms.Compose([transforms.ToTensor()])

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
