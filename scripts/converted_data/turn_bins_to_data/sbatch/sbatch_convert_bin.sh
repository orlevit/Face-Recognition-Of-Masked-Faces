#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120g
#SBATCH --gres=gpu
#SBATCH --job-name=run1
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --qos=gpu
#SBATCH --exclude=gpu6
echo "120g"
module load anaconda3 gcc/8.3.0 cuda/10.0.130 cudnn/7.6.5.32-10.2-linux-x64
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate tf_gpu_py36

python /home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/turn_bins_to_data/one_bin_to_data3.py -m 5
