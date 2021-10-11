import os
from easydict import EasyDict as edict

# ------------------------------------------------ miscellaneous ------------------------------------------------
# img2pose constants
DEPTH = 18
MAX_SIZE = 1400
MIN_SIZE = 600

# masks creation
FACE_MODEL_DENSITY = 0.001
LENS_RADIUS = 40 * FACE_MODEL_DENSITY
STRING_SIZE = 5
THRESHOLD_BUFFER = 30/100
HEAD_3D_NAME = "head3D"
EYE_MASK_NAME = "eyemask"
HAT_MASK_NAME = "hatmask"
SCARF_MASK_NAME = "scarfmask"
COVID19_MASK_NAME = "covid19mask"
SUNGLASSES_MASK_NAME = "sunglassesmask"
CENTER_FACE_PART = 'CENTER'
LEFT_FACE_PART = 'LEFT'
RIGHT_FACE_PART = 'RIGHT'
NEAR_NEIGHBOUR_STRIDE = 2
BBOX_REQUESTED_SIZE = 500
RANGE_CHECK = 30

#center
#img_size= [62,84,250,1888]
#kernel_size=[2,2,7.504,67]

#add
#img_size= [216]
#kernel_size=[3]

MIN_MASK_SIZE = 250
FILTER_MASK_RIGHT_POINT_IMAGE_SIZE = 1888
FILTER_SIZE_MASK_RIGHT_POINT = 67
FILTER_SIZE_MASK_ADD_LEFT_POINT = 4.021#(FILTER_DIM_MASK_ADD_LEFT_POINT, FILTER_DIM_MASK_ADD_LEFT_POINT)
FILTER_SIZE_MASK_ADD_RIGHT_POINT = 26

# Hat needs to be before the eyemask(even if only eyemask is needed, hat mask need to be included).
ALL_MASKS = f"{HAT_MASK_NAME},{EYE_MASK_NAME},{COVID19_MASK_NAME},{SUNGLASSES_MASK_NAME}"

# ------------------------------------------------ masks configuration ------------------------------------------------
config = {
    EYE_MASK_NAME: {
        "inds": {
            "left": 27086,
            "right": 27344,
            "top": 40454,
            "bottom": 8150
        },
        "add_forehead": True,
        "draw_rest_mask": True,
        "additional_masks_req": HAT_MASK_NAME,
        "filter_size": 9,
        "mask_front_points_calc": False,
        "mask_add_front_points_calc": False

    },
    HAT_MASK_NAME: {
        "inds": {
            "left": 20043,
            "right": 34895,
            "middle_bottom": 8094
        },
        "add_forehead": True,
        "draw_rest_mask": False,
        "additional_masks_req": None,
        "filter_size": 15,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": False,
        "mask_exists": False
    },
    SCARF_MASK_NAME: {
        "inds": {
            "left": 19564,
            "right": 34851,
            "middle_top": 8138
        },
        "add_forehead": False,
        "draw_rest_mask": False,
        "additional_masks_req": None,
        "filter_size": 15,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": False
    },
    COVID19_MASK_NAME: {
        "inds": {
            "center_middle": 25733,
            "right_middle": 178,
            "left_lower": 47948,
            "right_lower": 30393,#51439,
            "left_upper_string1": 16098,
            "left_upper_string2": 33284,
            "left_lower_string1": 29873, #20571, #51827,
            "left_lower_string2": 20567,
            "right_upper_string1": 185,
            "right_upper_string2": 21157,
            "right_lower_string1": 24714, #44003
            "right_lower_string2": 34082
        },# 29873
        "add_forehead": False,
        "draw_rest_mask": False,
        "additional_masks_req": None,
        "filter_size": 15,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": True
    },
    SUNGLASSES_MASK_NAME: {
        "inds": {
            "right": 4791
        },
        "add_forehead": False,
        "draw_rest_mask": False,
        "additional_masks_req": None,
        "filter_size": 9.355,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": True
    }
}
config = edict(config)

# ------------------------------------------------ Paths definitions ------------------------------------------------
# base directories
IMG2POSE_DIR = "."
CURRENT_DIR = "img2pose"
MODEL_DIR = "models"

# Paths to files
POSE_MEAN = os.path.join(IMG2POSE_DIR, CURRENT_DIR, MODEL_DIR, "WIDER_train_pose_mean_v1.npy")
POSE_STDDEV = os.path.join(IMG2POSE_DIR, CURRENT_DIR, MODEL_DIR, "WIDER_train_pose_stddev_v1.npy")
MODEL_PATH = os.path.join(IMG2POSE_DIR, CURRENT_DIR, MODEL_DIR, "img2pose_v1.pth")
VERTICES_PATH = os.path.join(IMG2POSE_DIR, CURRENT_DIR, "pose_references/vertices_trans.npy")
PATH_3D_POINTS = os.path.join(IMG2POSE_DIR, CURRENT_DIR, "pose_references/reference_3d_68_points_trans.npy")
