#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%A-%a.out
#SBATCH --error=%x-%A-%a.err
#SBATCH --qos=gpu
#SBATCH --gres=gpu

module load anaconda3
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate img2pose

python main.py  -i /RG/rg-tal/orlev/datasets/original_ds/AgeDB/agedb30 -o /home/orlev/work/Face-Recognition-Of-Masked-Faces/images_masked_crop/agedb30  > log_agedb30.txt
