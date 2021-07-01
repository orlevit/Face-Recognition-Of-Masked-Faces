#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --output=logs/slurm_logs/%x-%A-%a.out
#SBATCH --error=logs/slurm_logs/%x-%A-%a.err
#SBATCH --qos=gpu
#SBATCH --exclude=gpu6

module load anaconda3
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate $1

n=$SLURM_ARRAY_TASK_ID
echo "Testing $7"
shift_num=$((10 * ($n - 1)))
shift $shift_num

echo "python test_model_new_mask_and_nomask.py --data-dir-mask $2 --data-dir-nomask $3 --target-mask $4 --target-nomask $5--model $6 --roc-name $7 --threshold $8 | tail -3 |sed -n 1,2p >> $9"
#python test_model_new.py --data-dir $2 --target $3 --model $4 --roc-name $5 --threshold $6 | tail -3 |sed -n 1,2p >> $7

if [ $? -eq 0 ]; then
 echo "SUCCESS" >> $10
else
  echo "FAIL" >> $10
fi

