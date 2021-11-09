#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --output=%x-%A-%a.out
#SBATCH --error=%x-%A-%a.err
#SBATCH --qos=gpu

module load anaconda3
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate img2pose

python main_parallel.py  -i /RG/rg-tal/orlev/datasets/original_ds/CASIA-WebFace/ -o /home/orlev/work/Face-Recognition-Of-Masked-Faces/images_masked_crop/casia -cpu 48
