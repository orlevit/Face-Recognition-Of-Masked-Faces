import os

SLEEP_TIME = 30
MTCNN_ENV = 'mtcnn'
CASIA_PROPERTY_FILE = 'property'
ARCFACE_ENV = 'arcface'
LST_FILE = 'casia.lst'
LFW_PAIRS_FILE_NAME = 'lfw_pairs.txt'
CASIA_PAIRS_FILE_NAME = 'casia_pairs_5k.txt'
AGEDB30_PAIRS_FILE_NAME = 'agedb30_pairs.txt'
ARCFACE_DATSETS_LOC = '/home/orlev/work/project/insightface/datasets'
ARCFACE_DS_NAMES = ['nomask', 'eyemask', 'hatmask', 'covid19mask', 'sunglassesmask'] # 'scarfmask'
ARCFACE_VALIDATON_DS = 'casia.bin'
SCRIPTS_BASE_DIR = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts'
FINISHED_LOGS_DIR = os.path.join(SCRIPTS_BASE_DIR, 'logs', 'finished_logs')
SLURM_LOGS_DIR = os.path.join(SCRIPTS_BASE_DIR, 'logs', 'slurm_logs')
PREPARE_FILES_DIR = os.path.join(SCRIPTS_BASE_DIR, 'prepare_run/files/no_missings')
TEST_LOC = os.path.join(SCRIPTS_BASE_DIR, 'test')
NO_DATASET_TEST = 'org'

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
TEST_SAME_MEM = 8
TEST_SAME_JOBS_NAME = 'test_same'
TEST_SAME_COMMANDS_FILE = os.path.join(TEST_LOC, 'results_same_commands.txt')
TEST_SAME_RESULTS_FILE = os.path.join(TEST_LOC, 'results_same.csv')
TEST_SAME_TRACK_FILE = os.path.join(FINISHED_LOGS_DIR, 'test_same_finished.txt')
TEST_SAME_ROC = os.path.join(TEST_LOC, 'roc', 'same')
TEST_SAME_SBATCH_FILE = 'sbatch_test_same.sh'

# TEST MASK NOMASK
TEST_DIFF_MEM = 8 
TEST_DIFF_JOBS_NAME = 'test_diff'
TEST_DIFF_COMMANDS_FILE = os.path.join(TEST_LOC, 'results_diff_commands.txt')
TEST_DIFF_RESULTS_FILE = os.path.join(TEST_LOC, 'results_diff.csv')
TEST_DIFF_TRACK_FILE = os.path.join(FINISHED_LOGS_DIR, 'test_diff_finished.txt')
TEST_DIFF_ROC = os.path.join(TEST_LOC, 'roc', 'diff')
TEST_DIFF_SBATCH_FILE = 'sbatch_test_diff.sh'
##########################################################################

# files for idx and rec
CASIA_LST_FILE = os.path.join(PREPARE_FILES_DIR, LST_FILE)
CASIA_PROPERTY_LOC = os.path.join(PREPARE_FILES_DIR, CASIA_PROPERTY_FILE)

# pairs file
LFW_PAIRS = os.path.join(PREPARE_FILES_DIR, LFW_PAIRS_FILE_NAME)
AGEDB30_PAIRS = os.path.join(PREPARE_FILES_DIR, AGEDB30_PAIRS_FILE_NAME)
CASIA_PAIRS = os.path.join(PREPARE_FILES_DIR, CASIA_PAIRS_FILE_NAME)

# Models locaton
MODELS_BASE_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning'
NOMASK_DATA_LOC = os.path.join(ARCFACE_DATSETS_LOC, 'nomask')
NOMASK_MODEL_ORG_LOC = os.path.join(MODELS_BASE_LOC, 'r100-arcface-org_masked')
NOMASK_MODEL_LOC = os.path.join(MODELS_BASE_LOC, 'r100-arcface-no_masked')
EYE_MASK_MODEL = os.path.join(MODELS_BASE_LOC, 'r100-arcface-eye_masked')
COVID19_MASK_MODEL = os.path.join(MODELS_BASE_LOC, 'r100-arcface-covid19_masked')
HAT_MASK_MODEL = os.path.join(MODELS_BASE_LOC, 'r100-arcface-hat_masked')
#SCARF_MASK_MODEL = os.path.join(MODELS_BASE_LOC, 'r100-arcface-scarf_masked')
SUNGLASSES_MASK_MODEL = os.path.join(MODELS_BASE_LOC, 'r100-arcface-sunglasses_masked')

MODELS_DIRS_LIST = [NOMASK_MODEL_ORG_LOC, NOMASK_MODEL_LOC, EYE_MASK_MODEL, HAT_MASK_MODEL, COVID19_MASK_MODEL, SUNGLASSES_MASK_MODEL]

# Results table
RESULTS_HEADERS = 'Model, Masked Dataset, Threshold for best accuracy, Best accuracy on masked LFW, ' \
                  'AUC on masked LFW, Best accuracy on masked agedb30, AUC on masked agedb30'
RESULTS_TARGET_FILES = ['lfw', 'agedb30']
