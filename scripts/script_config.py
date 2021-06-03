import os
ALIGN_BASE_DIR = '/home/orlev/work/project/insightface_old/src/align'
DS_BASE_DIR = '/home/orlev/work/datasets/original_ds'

ALIGN_FUNC_ONLY = 'align_lfw_no_misc_or.py'
START_ENV = 'module load anaconda3; source activate mtcnn;'
ARCFACE_DATSETS_LOC = '/home/orlev/work/project/insightface/datasets'

# pairs file
LFW_PAIRS = os.path.join(DS_BASE_DIR, 'lfw/pairs.txt')
AGEDB30_PAIRS = os.path.join(DS_BASE_DIR, 'AgeDB/pairs.txt')
CASIA_PAIRS = os.path.join(DS_BASE_DIR, 'CASIA-WebFace/pairs_5k.txt')

# Functions
ALIGN_FUNC = os.path.join(ALIGN_BASE_DIR, ALIGN_FUNC_ONLY)
BIN_FUNC = '/home/orlev/work/project/insightface/src/data/lfw2pack.py'
