import os
LST_FILE ='CASIA-WebFace.lst'
PROPERTY = 'property'
ALIGN_BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/align'
SCRIPTS_BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts'
DS_BASE_DIR = '/home/orlev/work/datasets/original_ds'
CASIA_BASE_DIR = os.path.join(DS_BASE_DIR, 'CASIA-WebFace')
SLEEP_TIME = 10

ALIGN_FUNC_ONLY = 'align_lfw_no_misc_or.py'
MTCNN_ENV = 'mtcnn'
ARCFACE_ENV = 'arcface'
ARCFACE_DATSETS_LOC = '/home/orlev/work/project/insightface/datasets'

##################### SBATCHES #####################
SBATCH='sbatch --mem={}g --gres=gpu --job-name={} --array=1-{}' 

# ALIGN
ALIGN_MEM = 16
ALIGN_JOBS_NAME = 'align'
#ALIGN_JOBS_NUMBER = 5
ALIGN_FILE = os.path.join(SCRIPTS_BASE_DIR, 'align_finished.txt')
ALIGN_SBATCH_FILE = 'sbatch_align.sh'

# BIN
BIN_MEM = 16
BIN_JOBS_NAME = 'create_bin'
#BIN_JOBS_NUMBER = 5
BIN_FILE = os.path.join(SCRIPTS_BASE_DIR, 'bin_finished.txt')
BIN_SBATCH_FILE = 'sbatch_bin.sh'
#####################################################
#SBATCH='sbatch --mem={}g --gres=gpu --job-name={} --array=1-{} sbatch_bin.sh' 

# files for idx and rec
CASIA_LST_FILE = os.path.join(CASIA_BASE_DIR, LST_FILE)
CASIA_PROPERTY = os.path.join(CASIA_BASE_DIR, PROPERTY)

# pairs file
LFW_PAIRS = os.path.join(DS_BASE_DIR, 'lfw/pairs.txt')
AGEDB30_PAIRS = os.path.join(DS_BASE_DIR, 'AgeDB/agedb30/pairs.txt')
CASIA_PAIRS = os.path.join(CASIA_BASE_DIR, 'pairs_5k.txt')

# Functions
ALIGN_FUNC = os.path.join(ALIGN_BASE_DIR, ALIGN_FUNC_ONLY)
BIN_FUNC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/lfw2pack.py'
IDX_REC_FUNC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/face2rec2.py'
