import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import iglob

df = pd.DataFrame(columns=['Attempt', 'Loss type', 'optimizer', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'last_epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'test_loss', 'test_acc'])

LOGS_DIR = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_training/logs'
LOGS_FILES_LOC = os.path.join(LOGS_DIR, "*.out")
TABLE_FILE_LOC = os.path.join(LOGS_DIR, "table_adam.csv")

for i,file in tqdm(enumerate(iglob(LOGS_FILES_LOC), 1)):
    print('-->',file)
    if int(file.rsplit('-',1)[-1].split('.')[0]) < 89:
       with open(file) as f:
            lines = f.readlines()
     #  print('--------------------------------------')
     #  print(file)
     #  print(lines)
       # Parameters
       try:
          parameters = lines[2]
          splitted = parameters.split(',')
          lr = float(splitted[0].split('lr')[-1])
          momentum = splitted[1].split(':')[-1]
          dampening = splitted[2].split(':')[-1]
          weight_decay = splitted[3].split(':')[-1]
          nesterov = splitted[4].split(':')[-1].strip()
          
          # Split train-valid
          result_train_valid = lines[-9]#.split(':')
          splitted = re.split(':|/|\ |,', result_train_valid)
          epoch = float(splitted[1]) - 1
          train_loss = float(splitted[10])
          valid_loss = float(splitted[12])
          train_acc = float(splitted[18])
          valid_acc = float(splitted[20])
          
          # epoch_train
          if 1 < len(lines[-2].split('   '))  and lines[-2].split('   ')[1] == 'TEST':
              result_test = lines[-1]
              splitted = re.split(':|/|\ |\\n|,', result_test)
              test_loss = float(splitted[7])
              test_acc = float(splitted[13])
          else:
              test_loss = np.nan
              test_acc = np.nan
          df = df.append({'Attempt':i, 'Loss type':'MSE', 'optimizer':'SGD', 'lr':lr, 'momentum':momentum, 'dampening':dampening, 'weight_decay':weight_decay, 'nesterov':nesterov, 'last_epoch':epoch, 'train_loss':train_loss, 'valid_loss':valid_loss, 'train_acc':train_acc, 'valid_acc':valid_acc, 'test_loss':test_loss, 'test_acc':test_acc}, ignore_index=True)
       except Exception as e:
          print(f' This file didn"t processed well: {file}, {e}')

df.to_csv(TABLE_FILE_LOC)
