import os
ALIGN_BASE_DIR = '/home/orlev/work/project/insightface_old/src/align'
ALIGN_FUNC_ONLY = 'align_lfw_no_misc_or.py'
START_ENV = 'module load anaconda3; source activate mtcnn;'

ALIGN_FUNC = os.path.join(ALIGN_BASE_DIR, ALIGN_FUNC_ONLY)
