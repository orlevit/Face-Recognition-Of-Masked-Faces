import os
LST_FILE ='CASIA-WebFace.lst'
PROPERTY = 'property'
ALIGN_BASE_DIR = '/home/orlev/work/project/insightface_old/src/align'
DS_BASE_DIR = '/home/orlev/work/datasets/original_ds'
CASIA_BASE_DIR = os.path.join(DS_BASE_DIR, 'CASIA-WebFace')

ALIGN_FUNC_ONLY = 'align_lfw_no_misc_or.py'
START_ENV = 'module load anaconda3; source activate mtcnn;'
ARCFACE_DATSETS_LOC = '/home/orlev/work/project/insightface/datasets'

# files for idx and rec
CASIA_LST_FILE = os.path.join(CASIA_BASE_DIR, LST_FILE)
CASIA_PROPERTY = os.path.join(CASIA_BASE_DIR, PROPERTY)

# pairs file
LFW_PAIRS = os.path.join(DS_BASE_DIR, 'lfw/pairs.txt')
AGEDB30_PAIRS = os.path.join(DS_BASE_DIR, 'AgeDB/pairs.txt')
CASIA_PAIRS = os.path.join(CASIA_BASE_DIR, 'pairs_5k.txt')

# Functions
ALIGN_FUNC = os.path.join(ALIGN_BASE_DIR, ALIGN_FUNC_ONLY)
BIN_FUNC = '/home/orlev/work/project/insightface/src/data/lfw2pack.py'
IDX_REC_FUNC = '/home/orlev/work/project/insightface/src/data/face2rec2.py'