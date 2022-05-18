import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import iglob

df = pd.DataFrame(columns=['Attempt', '#train',  '#valid',  '#test',  'Loss type', 'optimizer', 'lr', 'beta1','beta2', 'weight_decay', 'last_epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'test_loss', 'test_acc'])

LOGS_DIR = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_training/logs/slurm'
LOGS_FILES_LOC = os.path.join(LOGS_DIR, "*.out")
TABLE_FILE_LOC = os.path.join(LOGS_DIR, "table_adam.csv")
LINE_DATASET = 6#3
LINE_OPTIM = 7#4
LINE_VALID_TRAIN_RESULTS = -9

for i,file in tqdm(enumerate(iglob(LOGS_FILES_LOC), 1)):
    if int(file.rsplit('-',1)[-1].split('.')[0]) < 89:
       with open(file) as f:
            lines = f.readlines()
     
       if 1 < len(lines) and lines[-1].startswith('Epoch'):
          # number of data
          parameters = lines[LINE_DATASET]
          splitted = parameters.split(',')
          train_samples = int(splitted[0].split(':')[-1])
          valid_samples = int(splitted[1].split(':')[-1])
          test_samples = int(splitted[2].split(':')[-1])
        # remove in the next text
          #train_samples = 61600
          #valid_samples = 15499
          #test_samples = 42000

          # Parameters
          parameters = lines[LINE_OPTIM]
          splitted = parameters.split(',')
          lr = float(splitted[0].split('lr')[-1])
          beta1 = splitted[1].split(':')[-1]
          beta2 = splitted[2].split(':')[-1]
          weight_decay = float(splitted[3].split(':')[-1])
          
          # Split train-valid
          result_train_valid = lines[LINE_VALID_TRAIN_RESULTS]#.split(':')
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
          df = df.append({'Attempt':i, '#train':train_samples, '#valid':valid_samples,'#test':test_samples, 'Loss type':'MSE', 'optimizer':'ADAM', 'lr':lr, 'beta1':beta1, 'beta2':beta2, 'weight_decay':weight_decay, 'last_epoch':epoch, 'train_loss':train_loss, 'valid_loss':valid_loss, 'test_loss':test_loss, 'train_acc':train_acc, 'valid_acc':valid_acc, 'test_acc':test_acc}, ignore_index=True)

df.to_csv(TABLE_FILE_LOC)
