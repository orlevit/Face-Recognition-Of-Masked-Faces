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
echo "Testing $5" >> $7
shift_num=$((8 * ($n - 1)))
shift $shift_num

echo "python test_model_new.py --data-dir $2 --target $3 --model $4 --roc-name $5 --threshold $6 | tail -3 |sed -n 1,2p| xargs -0 -I % echo -e '$5 \n%'" >> $7
python test_model_new.py --data-dir $2 --target $3 --model $4 --roc-name $5 --threshold $6 | tail -3 |sed -n 1,2p| xargs -0 -I % echo -e "$5\n$6\n%"

if [ $? -eq 0 ]; then
 echo "SUCCESS" >> $8
else
  echo "FAIL" >> $8
fi
