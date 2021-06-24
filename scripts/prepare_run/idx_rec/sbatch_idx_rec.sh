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
echo "Job array index >>> $n <<< started to work"
shift_num=$((4 * ($n - 1)))
shift $shift_num

cd $2
python /home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/idx_rec/face2rec2.py $2 
cp *.idx "$3/train.idx"
cp *.rec "$3/train.rec"

if [ $? -eq 0 ]; then
 echo "SUCCESS" >> $4
else
  echo "FAIL" >> $4
fi

