#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --job-name=make_ijbc_embs
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --output=/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_testing/threshold_tests/ijbc/logs/%x-%A-%a.out
#SBATCH --error=/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_testing/threshold_tests/ijbc/logs/%x-%A-%a.err
#SBATCH --qos=gpu
#SBATCH --array=1-5%3
#SBATCH --exclude=gpu6

module load anaconda3 gcc/8.3.0 cuda/10.0.130 cudnn/7.6.5.32-10.2-linux-x64
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate com

n=$SLURM_ARRAY_TASK_ID

python /home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_testing/threshold_tests/ijbc/make_img_embs.py --split $n
