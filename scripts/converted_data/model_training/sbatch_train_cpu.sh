#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5g
#SBATCH --job-name=train_search
#SBATCH --output=/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_training/logs/slurm/%x-%A-%a.out
#SBATCH --error=/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_training/logs/slurm/%x-%A-%a.err
#SBATCH --qos=gpu
#SBATCH --array=1-1

# %A = job_name
# %a = array id

#conda deactivate
module load anaconda3 gcc/8.3.0 cuda/10.0.130 cudnn/7.6.5.32-10.2-linux-x64
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate com

comb_number=$SLURM_ARRAY_TASK_ID
echo "Job array index >>> $comb_number <<< started to work"
echo $(date)
python /RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_training/main.py -cn $comb_number
