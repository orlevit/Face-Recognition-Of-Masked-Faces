import os
from easydict import EasyDict as edict

# ------------------------------------------------ miscellaneous ------------------------------------------------
SLOPE_TRAPEZOID = -0.01
INTERCEPT_TRAPEZOID = 1.2
MIN_TRAPEZOID_INPUT = 20
MIN_TRAPEZOID_OUTPUT = 0.1
YAW_IMPORTANCE = 0.8
PITCH_IMPORTANCE = 0.2
MIN_POSE_SCORES = 0.8

# img2pose constants
DEPTH = 18
MAX_SIZE = 1400
MIN_SIZE = 600

# masks creation
FACE_MODEL_DENSITY = 0.001
LENS_RADIUS = 40 * FACE_MODEL_DENSITY
STRING_SIZE = 5
HATMASK_EYEMASK_GAP_FILLER = 5
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
MIN_MASK_PIXELS = 10
MIN_DETECTED_FACE_PERCENTAGE = 0.01
MASK_EXTEND_PIXELS = 13
MASK_EXTEND_BBOX_NORM = 150
MIN_POSE_OPEN_EYEMASK = 0.8
#center
#img_size= [62,84,250,1888]
#kernel_size=[2,2,7.504,67]

#add
#img_size= [216]
#kernel_size=[3]

MIN_MASK_SIZE = 250
FILTER_MASK_RIGHT_POINT_IMAGE_SIZE = 1888
MASK_RIGHT_POINT = 67
ADD_LEFT_POINT = 4.021#(FILTER_DIM_MASK_ADD_LEFT_POINT, FILTER_DIM_MASK_ADD_LEFT_POINT)
ADD_RIGHT_POINT = 26

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
        "mask_filter_size": 15,
        "rest_filter_size": 9,
        "mask_front_points_calc": False,
        "mask_add_front_points_calc": False,
        "main_mask_components_number": 1

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
        "mask_filter_size": 15,
        "rest_filter_size": 15,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": False,
        "main_mask_components_number": 1
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
        "mask_filter_size": 15,
        "rest_filter_size": 15,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": False,
        "main_mask_components_number": 1
    },
    COVID19_MASK_NAME: {
        "inds": {
            "center_middle": 25733,
            "right_middle": 178,
            "left_lower": 47948,
            "right_lower": 30393,
            "left_upper_string1": 15583,# 16098
            "left_upper_string2": 33284,# smaller upper string 31481,
            "left_lower_string1": 29488,# 29873
            "left_lower_string2": 20567,
            "right_upper_string1": 702, #185
            "right_upper_string2": 21157,# smaller upper string 22967 ,
            "right_lower_string1": 25103,#24714,
            "right_lower_string2": 34082
        },# 29873
        "add_forehead": False,
        "draw_rest_mask": False,
        "additional_masks_req": None,
        "mask_filter_size": 15,
        "rest_filter_size": 15,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": True,
        "main_mask_components_number": 1
    },
    SUNGLASSES_MASK_NAME: {
        "inds": {
            "center_right_lens": 4791,
            "right_lens_right_side": 3109,
            "left_lens_left_side": 12831
        },
        "add_forehead": False,
        "draw_rest_mask": False,
        "additional_masks_req": None,
        "mask_filter_size": 9.355,
        "rest_filter_size": 9.355,
        "mask_front_points_calc": True,
        "mask_add_front_points_calc": True,
        "main_mask_components_number": 2
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


# -------------------------------------------- Top eyemask head indices ----------------------------------------------

TOP_EYEMASK_INDS = [38927, 38966, 39042, 39080, 39154, 39191, 39227, 39262, 39297,
                    39366, 39400, 39434, 39467, 39500, 39533, 39565, 39597, 39629,
                    39660, 39691, 39722, 39752, 39753, 39782, 39812, 39841, 39870,
                    39899, 39927, 39955, 39983, 40011, 40039, 40066, 40093, 40094,
                    40120, 40147, 40174, 40201, 40202, 40228, 40254, 40255, 40280,
                    40306, 40332, 40358, 40384, 40410, 40436, 40462, 40488, 40514,
                    40541, 40540, 40566, 40592, 40618, 40644, 40670, 40697, 40724,
                    40751, 40778, 40805, 40833, 40832, 40859, 40887, 40915, 40943,
                    40971, 40999, 41028, 41057, 41088, 41086, 41116, 41146, 41176,
                    41207, 41238, 41270, 41269, 41301, 41366, 41365, 41431, 41464,
                    41498, 41532, 41566, 41636, 41671, 41707, 41744, 41818, 41856,
                    39004, 39117, 39332, 41333, 41398, 41601, 41781, 41894, 41932,
                    41933, 41972, 41819, 41708, 38887, 38888, 38928, 39043]