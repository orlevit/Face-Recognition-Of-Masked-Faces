import os

SLEEP_TIME = 30
MTCNN_ENV = 'mtcnn'
CASIA_PROPERTY = 'property'
ARCFACE_ENV = 'arcface'
LST_FILE = 'casia.lst'
LFW_PAIRS_FILE_NAME = 'lfw_pairs.txt'
CASIA_PAIRS_FILE_NAME = 'casia_pairs_5k.txt'
AGEDB30_PAIRS_FILE_NAME = 'agedb30_pairs.txt'
# ALIGN_FUNC_ONLY = 'align_lfw_no_misc_or.py'
# DS_BASE_DIR = '/home/orlev/work/datasets/original_ds'
# CASIA_BASE_DIR = os.path.join(DS_BASE_DIR, 'CASIA-WebFace')
# ALIGN_BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/align'
ARCFACE_DATSETS_LOC = '/home/orlev/work/project/insightface/datasets'
ARCFACE_DS_NAMES = ['eyemask', 'hatmask', 'sunglassesmask', 'scarfmask', 'coronamask']
ARCDACE_VALIDATON_DS = 'casia.bin'
SCRIPTS_BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts'
FINISHED_LOGS_DIR = os.path.join(SCRIPTS_BASE_DIR,'logs','finished_logs')
SLURM_LOGS_DIR = os.path.join(SCRIPTS_BASE_DIR,'logs','slurm_logs')
PREPARE_FILES_DIR = os.path.join(SCRIPTS_BASE_DIR, 'prepare_run/files/no_missings')

################################ SBATCHES ################################
SBATCH = 'sbatch --mem={}g --gres=gpu --job-name={} --array=1-{}'

# ALIGN
ALIGN_MEM = 16
ALIGN_JOBS_NAME = 'align'
ALIGN_FILE = os.path.join(FINISHED_LOGS_DIR, 'align_finished.txt')
ALIGN_SBATCH_FILE = 'prepare_run/align/sbatch_align.sh'

# BIN
BIN_MEM = 16
BIN_JOBS_NAME = 'create_bin'
BIN_FILE = os.path.join(FINISHED_LOGS_DIR, 'bin_finished.txt')
BIN_SBATCH_FILE = 'prepare_run/bin/sbatch_bin.sh'

# IDX_REC
IDX_REC_MEM = 16
IDX_REC_JOBS_NAME = 'create_idx_rec'
IDX_REC_FILE = os.path.join(FINISHED_LOGS_DIR, 'idx_rec_finished.txt')
IDX_REC_SBATCH_FILE = 'prepare_run/idx_rec/sbatch_idx_rec.sh'

# TEST SAME
TEST_MEM = 16
TEST_JOBS_NAME = 'test_same'
TEST_RESULTS_FILE = os.path.join(SCRIPTS_BASE_DIR, 'test/results_same.txt')
TEST_TRACK_FILE = os.path.join(FINISHED_LOGS_DIR , 'test_same_finished.txt')
TEST_SBATCH_FILE = 'test/sbatch_test_same.sh'
##########################################################################

# files for idx and rec
CASIA_LST_FILE = os.path.join(PREPARE_FILES_DIR, LST_FILE)
CASIA_PROPERTY = os.path.join(PREPARE_FILES_DIR, CASIA_PROPERTY)

# pairs file
LFW_PAIRS = os.path.join(PREPARE_FILES_DIR, LFW_PAIRS_FILE_NAME)
AGEDB30_PAIRS = os.path.join(PREPARE_FILES_DIR, AGEDB30_PAIRS_FILE_NAME)
CASIA_PAIRS = os.path.join(PREPARE_FILES_DIR, CASIA_PAIRS_FILE_NAME)
