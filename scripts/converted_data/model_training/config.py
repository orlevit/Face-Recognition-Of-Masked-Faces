import os
import numpy as np

# Prev models charactaristics
MODELS_NUMBER = 7
EMBBEDINGS_NUMBER = 512
EMBBEDINGS_REDUCED = 32
LINEAR_INIT = 512
BILINEAR_INIT =  32 * 7

# What kind of ds to load and stat
TRAIN_DS_IND = 0
VALID_DS_IND = 1
TEST_DS_IND = 2
SPLIT_TRAIN = 0.8

# Training hyperparameters
EPOCHS = 1000000
MIN_LOSS_SAVE = 1
EARLY_STOP_DIFF = float('inf')
WHOLE_DATA_BATCH = -1 # contant that meant to take the batch size as whole the dataset
BATCH_SIZE = 1000

# Locations
BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data'
RUN_DIR = os.path.join(BASE_DIR, 'model_training/logs/run')
#TRAIN_DATA_LOC = os.path.join(BASE_DIR, 'all_train/all_data.pt')
#TRAIN_LABELS_LOC = os.path.join(BASE_DIR, 'all_train/all_labels.pt')
TRAIN_DATA_LOC = os.path.join('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/all', 'data.npy')
TRAIN_LABELS_LOC = os.path.join('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/all', 'labels.npy')
TEST_DATA_LOC = os.path.join(BASE_DIR, 'lfw_new_composed_diff_masks/all_test/all_data.pt')
TEST_LABELS_LOC = os.path.join(BASE_DIR, 'lfw_new_composed_diff_masks/all_test/all_labels.pt')
#TEST_DATA_LOC = os.path.join(BASE_DIR, 'ready_data/77000_test_lfw_casia_pairs/all_data.pt')
#TEST_LABELS_LOC = os.path.join(BASE_DIR, 'ready_data/77000_test_lfw_casia_pairs/all_labels.pt')
MODELS_SAVE_PATH = os.path.join(BASE_DIR, 'models')

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
