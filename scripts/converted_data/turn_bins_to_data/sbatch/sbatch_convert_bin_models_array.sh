#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120g
#SBATCH --gres=gpu
#SBATCH --job-name=run
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --qos=gpu
#SBATCH --array=4-7
#SBATCH --exclude=gpu6

# %A = job_name
# %a = array id

#conda deactivate
module load anaconda3 gcc/8.3.0 cuda/10.0.130 cudnn/7.6.5.32-10.2-linux-x64
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate tf_gpu_py36

n=$SLURM_ARRAY_TASK_ID
model_num=$(($n - 1))
echo "Job array index >>> $n <<< started to work"
python /home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/turn_bins_to_data/one_bin_to_data3.py -m $model_num
