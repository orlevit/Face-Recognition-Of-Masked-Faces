#!/bin/bash

### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --job-name=create_bin
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err
#SBATCH --qos=gpu

#conda deactivate
#gcc/8.3.0 
module load anaconda3 
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source activate tf_gpu_py36_4

cd /home/orlev/work/Face-Recognition-Of-Masked-Faces
python /home/orlev/work/project/insightface_old/src/data/lfw2pack.py --data-dir $1 --output $2 --image-size 112,112
