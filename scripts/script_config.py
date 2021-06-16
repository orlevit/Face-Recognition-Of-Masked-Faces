import os

SLEEP_TIME = 10
MTCNN_ENV = 'mtcnn'
PROPERTY = 'property'
ARCFACE_ENV = 'arcface'
LST_FILE = 'CASIA-WebFace.lst'
ALIGN_FUNC_ONLY = 'align_lfw_no_misc_or.py'
DS_BASE_DIR = '/home/orlev/work/datasets/original_ds'
CASIA_BASE_DIR = os.path.join(DS_BASE_DIR, 'CASIA-WebFace')
ARCFACE_DATSETS_LOC = '/home/orlev/work/project/insightface/datasets'
SCRIPTS_BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts'
ALIGN_BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/align'

################################ SBATCHES ################################
SBATCH = 'sbatch --mem={}g --gres=gpu --job-name={} --array=1-{}'

# ALIGN
ALIGN_MEM = 16
ALIGN_JOBS_NAME = 'align'
ALIGN_FILE = os.path.join(SCRIPTS_BASE_DIR, 'align_finished.txt')
ALIGN_SBATCH_FILE = 'prepare_run/align/sbatch_align.sh'

# BIN
BIN_MEM = 16
BIN_JOBS_NAME = 'create_bin'
BIN_FILE = os.path.join(SCRIPTS_BASE_DIR, 'bin_finished.txt')
BIN_SBATCH_FILE = 'prepare_run/bin/sbatch_bin.sh'

# IDX_REC
IDX_REC_MEM = 16
IDX_REC_JOBS_NAME = 'create_idx_rec'
IDX_REC_FILE = os.path.join(SCRIPTS_BASE_DIR, 'idx_rec_finished.txt')
IDX_REC_SBATCH_FILE = 'prepare_run/idx_rec/sbatch_idx_rec.sh'
##########################################################################

# files for idx and rec
CASIA_LST_FILE = os.path.join(CASIA_BASE_DIR, LST_FILE)
CASIA_PROPERTY = os.path.join(CASIA_BASE_DIR, PROPERTY)

# pairs file
LFW_PAIRS = os.path.join(DS_BASE_DIR, 'lfw/pairs.txt')
AGEDB30_PAIRS = os.path.join(DS_BASE_DIR, 'AgeDB/agedb30/pairs.txt')
CASIA_PAIRS = os.path.join(CASIA_BASE_DIR, 'pairs_5k.txt')

## Functions
# ALIGN_FUNC = os.path.join(ALIGN_BASE_DIR, ALIGN_FUNC_ONLY)
# BIN_FUNC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/lfw2pack.py'
# IDX_REC_FUNC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/face2rec2.py'
