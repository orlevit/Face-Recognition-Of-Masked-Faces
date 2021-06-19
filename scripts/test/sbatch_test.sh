#!/bin/bash
### Allocation Start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --qos=gpu

module load anaconda3
source /opt/apps/anaconda3/etc/profile.d/conda.sh
source deactivate
source activate $1

n=$SLURM_ARRAY_TASK_ID
echo "Job array index >>> $n <<< started to work"
shift_num=$((4 * ($n - 1)))
shift $shift_num

python test_model_new.py --data-dir $2 --target $3 --model $4 --roc-name $5 --threshold $6 | tail -3 |sed -n 1,2p

if [ $? -eq 0 ]; then
 echo "SUCCESS" >> $7
else
  echo "FAIL" >> $7
fi

