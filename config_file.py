import os
from easydict import EasyDict as edict

# ------------------------------------------------ masks configuration ------------------------------------------------
config = {
    "eyemask": {
        "inds": {
            "left": 27086,
            "right": 27344,
            "top": 40454,
            "bottom": 8150
        },
        "add_forehead": True,
        "draw_rest_mask": True
    },
    "hatmask": {
        "inds": {
            "left": 20043,
            "right": 34895,
            "middle_bottom": 8094
        },
        "add_forehead": True,
        "draw_rest_mask": True
    },
    "scarfmask": {
        "inds": {
            "left": 19564,
            "right": 34851,
            "middle_top": 8138
        },
        "add_forehead": False,
        "draw_rest_mask": True
    },
    "coronamask": {
        "inds": {
            "center_middle": 25733,
            "right_middle": 178,
            "left_lower": 47948,
            "right_lower": 51439,
            "left_upper_string1": 16098,
            "left_upper_string2": 33284,
            "left_lower_string1": 51827,
            "left_lower_string2": 20567,
            "right_upper_string1": 185,
            "right_upper_string2": 21157,
            "right_lower_string1": 44003,
            "right_lower_string2": 34082
        },
        "add_forehead": False,
        "draw_rest_mask": False
    }
}

config = edict(config)

# EYE_MASK_IND = [27086, 27344, 40454, 8150]
# HAT_MASK_IND = [20043, 34895, 8094]
# SCARF_MASK_IND = [19564, 34851, 8138]
# CORONA_MASK_IND = [16091, 25733, 178, 47948, 51439, 16098, 33284, 51827, 20567, 185, 21157, 44003, 34082]

# EYE_MASK = 'eyeMask'
# HAT_MASK = 'hatMask'
# SCARF_MASK = 'scarfMask'
# CORONA_MASK = 'coronaMask'
# MASKS_NAMES = [EYE_MASK, HAT_MASK, SCARF_MASK, CORONA_MASK]

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

# ------------------------------------------------ miscellaneous ------------------------------------------------
# img2pose constants
THRESHOLD = 0.9
DEPTH = 18
MAX_SIZE = 1400
MIN_SIZE = 600

# masks creation
MORPHOLOGICAL_CLOSE_FILTER = (15, 15)
FACE_MODEL_DENSITY = 0.001
STRING_SIZE = 5
HAT_MASK_CONFIG_NAME = "hatmask"
