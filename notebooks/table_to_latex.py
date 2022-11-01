import os
import pdb
import numpy as np
import pandas as pd

ROUNDING = 3
NOT_TO_PROCESS = 'NO'
BASE_LOC = r'/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/test'
SAME_FILE = os.path.join(BASE_LOC, 'results_same.csv')
DIFF_FILE = os.path.join(BASE_LOC, 'results_diff.csv')

# Choose SAME_FILE or DIFF_FILE
the_file = pd.read_csv(SAME_FILE)
###############################

def change_model_type_name(name):
    if name == 'org':
       return 'No mask'
    if name == 'eye':
       return 'Eye mask'
    if name == 'covid19':
       return 'Covid19 mask'
    if name == 'hat':
       return 'Hat mask'
    if name == 'Sunglasses':
       return 'Sunglasses mask'
    if name in ['no', 'sh', 'sc']:
        return NOT_TO_PROCESS
   # if name == 'sh':
   #    return 'Sunglasses+Hat mask'
   # if name == 'sc':
   #    return 'Sunglasses+Covid19 mask'
    return name

def change_images_type_name(name):
    if name == 'shmask':
       return 'Sunglasses+Hat'
    if name == 'scmask':
       return 'Sunglasses+Covid19'
    return name.replace('mask', '').capitalize()

def make_row(row_in_latex, row, max_dict, row_idx):
    for col_idx in range(3, 7):
        row_in_latex += r' & \centering '
        value = splitter.join(['{:.3}'.format(round(float(term), ROUNDING)) for term in row[col_idx].split(splitter)])
        if col_idx in max_dict and row_idx in max_dict[col_idx]:
           row_in_latex += '\\bf{' + value + '}'
        else:
           row_in_latex += value
    return row_in_latex

        
def split_df(df, split):
    def make_dict(max_dict, values, col_idx):
        max_val = max(values)
        max_values_loc = [ i for i in range(len(values)) if values[i] == max_val ]
        max_dict[col_idx] = max_values_loc
        #max_val = np.argsort(values)[-1]
        #max_dict[col_idx] = max_val 
    max_dict = {}
    selected_split_df = df.iloc[7*(split - 1) : 7*split].iloc[:5]
    for col_idx in range(3, 7):
        values = [round(float(i.split('+-')[0]), ROUNDING) for i in np.squeeze(selected_split_df.iloc[:, [col_idx]].to_numpy())]
        make_dict(max_dict, values, col_idx)
    return selected_split_df, max_dict

row_in_latex = ''
splitter = '+-'
splits_number = int(len(the_file) / 7)
for split in range(1, splits_number + 1):
    first_row_ind = True
    print(the_file.iloc[(split-1)*7][0], the_file.iloc[(split-1)*7][2])
    model_name = change_model_type_name(the_file.iloc[(split-1)*7][0])
    threshold = str(the_file.iloc[(split-1)*7][2])
    if model_name != NOT_TO_PROCESS:
       row_in_latex += f'\centering\multirow{{5}}{{*}}' + '{' +model_name+'}'+' & ' +f'\centering\multirow{{5}}{{*}}' +'{'+threshold+'}'
    
       selected_split_df, max_dict = split_df(the_file, split)
       for row_i, (idx, row) in enumerate(selected_split_df.iterrows()):
           if row[0] != 'no':
               if first_row_ind:
                  first_char = ' & '
               else:
                  first_char = ' && '
               row_in_latex += first_char + change_images_type_name(row[1])
               row_in_latex = make_row(row_in_latex, row, max_dict, row_i)
               first_row_ind = False
               row_in_latex += r' \tabularnewline \cline{3-7} '
               #row_in_latex += r' \\ \cline{3-7} '
       row_in_latex += ' \hline '
print(row_in_latex)
