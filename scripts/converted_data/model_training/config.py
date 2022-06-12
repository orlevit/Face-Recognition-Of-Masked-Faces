import os
import numpy as np

# Prev models charactaristics
MODELS_NUMBER = 7
EMBBEDINGS_NUMBER = 512
EMBBEDINGS_REDUCED = 32
LINEAR_INIT = 512
BILINEAR_INIT =  32 * 7
LOAD_MODEL = False # Is to load pre-existing model. from PRELOADED_MODEL_LOC

# What kind of ds to load and stat
TRAIN_DS_IND = 0
VALID_DS_IND = 1
TEST_DS_IND = 2
SPLIT_TRAIN = 0.8

# Training hyperparameters
EPOCHS = 1000000
MIN_LOSS_SAVE = 0.5
EARLY_STOP_DIFF = float('inf')
WHOLE_DATA_BATCH = -1 # contant that meant to take the batch size as whole the dataset
BATCH_SIZE = 1000
THRESHOLDS_INTERVALS = 0.005

# Locations
BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data'
RUN_DIR = os.path.join(BASE_DIR, 'model_training/logs/run')
TEST_DATA_LOC = os.path.join(BASE_DIR, 'lfw_new_composed_diff_masks/all_test/all_data.pt')
TEST_LABELS_LOC = os.path.join(BASE_DIR, 'lfw_new_composed_diff_masks/all_test/all_labels.pt')
MODELS_SAVE_PATH = os.path.join(BASE_DIR, 'models')

# combinedV1
#TRAIN_DATA_LOC = os.path.join('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV1/all', 'data.pt')
#TRAIN_LABELS_LOC = os.path.join('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV1/all', 'labels.pt')
#PRELOADED_MODEL_LOC = os.path.join(MODELS_SAVE_PATH, '350000_pairs_batch_all_hidden4096_NeuralNetwork5_lr1e-05_32_D20_02_2022_T18_53_58_770221.pt')
# combinedV2
TRAIN_DATA_LOC = os.path.join('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV2/all/350k_pairs', 'data.pt')
TRAIN_LABELS_LOC = os.path.join('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV2/all/350k_pairs', 'labels.pt')
PRELOADED_MODEL_LOC = os.path.join(MODELS_SAVE_PATH, '300000_pairs_same_masks_hidden4096_NeuralNetwork5_lr1e-07_32_D30_05_2022_T16_54_35_404599.pt')

# hyperparameters search
#hsdict = {'lr': [np.power(10.0, -6),np.power(10.0, -7),np.power(10.0, -8)],\
hsdict = {'lr': [np.power(10.0, -7)]}#,np.power(10.0, -3),5*np.power(10.0, -4),np.power(10.0, -5)]}
#hsdict = {'lr': [np.power(10.0, -6)]}#,np.power(10.0, -5),np.power(10.0, -7),np.power(10.0, -8)]}
#          'beta1': [0.9, 0.85, 0.95], \
#          'beta2': [0.999, 0.99, 0.98], \
#          'weight_decay': [0.0, 0.5, 0.9]}
#hsdict = {'lr': [np.power(10.0, -5),np.power(10.0, -6),np.power(10.0, -7)],\
#          'momentum': [0.0, 0.5, 0.9], \
#          'dampening': [0.0, 0.5, 0.9], \
#          'weight_decay': [0.0, 0.5, 0.9], \
#          'nesterov': [False, True]}
