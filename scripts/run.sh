#!/bin/sh
echo 'Start running Insightface on all masks'

source /home/orlev/work/project/scripts/run/constants.sh

for ((j=0; j<${#name_executed_arr[@]}; j++ ));
do
	sbatch /home/orlev/work/project/scripts/run/run_one.sh $j
done

python job.py

if [ $? -eq 0 ]; then
 echo "SUCCESS" > outputfile
else
  echo "FAIL" > outputfile
fi
