#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --gres=gpu
#SBATCH --job-name=run
#SBATCH --output=logs/slurm_logs/%x-%A-%a.out
#SBATCH --error=logs/slurm_logs/%x-%A-%a.err
#SBATCH --qos=gpu
#SBATCH --array=1-5

# %A = job_name
# %a = array id

#conda deactivate
module load anaconda3 gcc/8.3.0 cuda/10.0.130 cudnn/7.6.5.32-10.2-linux-x64
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source activate tf_gpu_py36

n=$SLURM_ARRAY_TASK_ID
name=$(sed "${n}q;d" input_tests.txt)

echo "Job array index >>> $n <<< started to work"

python -u  /home/orlev/work/project/insightface/recognition/ArcFace/train.py --network r100 --loss arcface --dataset $name
