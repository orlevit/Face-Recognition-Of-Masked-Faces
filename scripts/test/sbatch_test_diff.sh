#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --output=/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/logs/slurm_logs/%x-%A-%a.out
#SBATCH --error=/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/logs/slurm_logs/%x-%A-%a.err
#SBATCH --qos=gpu
#SBATCH --exclude=gpu6

module load anaconda3
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate $1

n=$SLURM_ARRAY_TASK_ID
echo "Testing $6"
shift_num=$((9 * ($n - 1)))
shift $shift_num

echo "python test_model_new_mask_and_nomask.py --data-dir-mask $2 --data-dir-nomask $3 --target-mask $4 --target-nomask $4 --model $5 --roc-name $6 --threshold $7 | tail -3 |sed -n 1,2p | xargs -0 -I % echo -e '$6 \n%' >> $8"
python test_model_new_mask_and_nomask.py --data-dir-mask $2 --data-dir-nomask $3 --target-mask $4 --target-nomask $4 --model $5 --roc-name $6 --threshold $7 | tail -3 |sed -n 1,2p | xargs -0 -I % echo -e "$6 \n%" >> $8

if [ $? -eq 0 ]; then
 echo "SUCCESS" >> $9
else
  echo "FAIL" >> $9
fi
