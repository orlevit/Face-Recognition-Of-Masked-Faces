import os
import pandas as pd

BASE_IJBC_DIR = '/home/orlev/work/downloads/IJB/IJB/IJB-C'
BASE_TRANS_MODEL_DIR = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning'
BASE_EMBS_FILES_DIR  = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_testing/threshold_tests/ijbc'
IMG_EMB_ARR = os.path.join(BASE_EMBS_FILES_DIR, 'data_files', 'ie_arr_{}.npy')
IMG_IDX_ARR = os.path.join(BASE_EMBS_FILES_DIR, 'data_files', 'ie_idx_{}.npy')
IMG_EMB_ARR_ALL = os.path.join(BASE_EMBS_FILES_DIR, 'data_files', 'ie_arr_all.npy')
IMG_IDX_ARR_ALL = os.path.join(BASE_EMBS_FILES_DIR, 'data_files', 'ie_idx_all.npy')
#IMAGES_DIR = os.path.join(BASE_IJBC_DIR, 'ijbc_test', '{}')
#BIN_LOC_SKELETON = '/RG/rg-tal/orlev/datasets/original_ds/MFR2_bg/a{}mask/a{}mask.bin'
MODEL_DIR_LOC_SKELETON = os.path.join(BASE_TRANS_MODEL_DIR, 'r100-arcface-{}_masked')

IMAGES_DIR = os.path.join(BASE_IJBC_DIR, 'ijbc_test')
#IJBC_MATCHES_LOC = os.path.join(BASE_IJBC_DIR, 'protocols/test2', 'match.csv')
#IJBC_TEMPLATE_LOC = os.path.join(BASE_IJBC_DIR, 'protocols/test2', 'enroll_templates.csv')
IJBC_MATCHES_LOC = os.path.join(BASE_IJBC_DIR, 'protocols/test1', 'match.csv')
IJBC_TEMPLATE_LOC = os.path.join(BASE_IJBC_DIR, 'protocols/test1', 'joined_verif_enroll.csv')


BASE_COMPOSED_MODELS_PATH = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/models'
COMPOSED_MODEL_PATH = os.path.join(BASE_COMPOSED_MODELS_PATH, '20000pairsV1_reduce8_seed42_NeuralNetwork15_lastHidden4096_lr1e-05_32_D27_06_2022_T19_34_34_794674.pt')

IMAGE_SIZE = (112, 112)
MAKE_EMB_SPLIT_NUM = 5
SPLIT_NUM = 1000
#BATCH_SIZE = MATCHES_SIZE // SPLIT_NUM + 1

MATCH_DF = pd.read_csv(IJBC_MATCHES_LOC)
TEMPLATE_DF = pd.read_csv(IJBC_TEMPLATE_LOC)
