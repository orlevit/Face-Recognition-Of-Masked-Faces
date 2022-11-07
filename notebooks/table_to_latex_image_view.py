import os
import pdb
import numpy as np
import pandas as pd

ROUNDING = 3
NOT_TO_PROCESS = 'NO'
ORDER_DB={'nomask':1, 'eyemask':2, 'hatmask':3,'covid19mask':4,'sunglassesmask':5,'shmask':6,'scmask':7}
ORDER_MODEL={'org':1, 'eye':2, 'hat':3,'covid19':4,'sunglasses':5,'sh':6,'sc':7, 'no':8}
BASE_LOC = r'/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/test'
SAME_FILE = os.path.join(BASE_LOC, 'results_same.csv')
DIFF_FILE = os.path.join(BASE_LOC, 'results_diff.csv')

# Choose SAME_FILE or DIFF_FILE
the_file = pd.read_csv(DIFF_FILE)
###############################

def change_db_type_name(name):
    if name == 'nomask':
       return 'No mask'
    if name == 'eyemask':
       return 'Eye mask'
    if name == 'covid19mask':
       return 'Covid19 mask'
    if name == 'hatmask':
       return 'Hat mask'
    if name == 'sunglassesmask':
       return 'Sunglasses mask'
    if name in ['shmask', 'scmask']:
        return NOT_TO_PROCESS
    return name

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
    return name

def change_images_type_name(name):
    if name == 'shmask':
       return 'Sunglasses+Hat'
    if name == 'scmask':
       return 'Sunglasses+Covid19'
    return name.replace('mask', '').capitalize()

def make_row(row_in_latex, row, max_dict, row_idx):
    for col_idx in range(2, 7):
        row_in_latex += r' & \centering '
        value = splitter.join(['{:.3f}'.format(float(term)) for term in row[col_idx].split(splitter)])
        if col_idx in max_dict and row_idx in max_dict[col_idx]:
           row_in_latex += '\\bf{' + value + '}'
        else:
           row_in_latex += value

    row_in_latex += ' & \centering '  + str(int(row.iloc[-1]))

    return row_in_latex

def get_splitted_srd_or_mean(lst, num):
    return [i.split(splitter)[num] for i in  lst]

def add_columns(df): 
    lfw_auc = df.iloc[:, [3]].values.squeeze().tolist()
    agedb30_auc = df.iloc[:, [5]].values.squeeze().tolist()

    lfw_mean  = get_splitted_srd_or_mean(lfw_auc, 0) 
    lfw_std = get_splitted_srd_or_mean(lfw_auc, 1) 

    agedb30_mean  = get_splitted_srd_or_mean(agedb30_auc, 0) 
    agedb30_std = get_splitted_srd_or_mean(agedb30_auc, 1) 


    lfw_mean = np.expand_dims(np.asarray(lfw_mean, dtype=float),axis=1)
    lfw_std = np.expand_dims(np.asarray(lfw_std, dtype=float),axis=1)

    agedb30_mean = np.expand_dims(np.asarray(agedb30_mean, dtype=float),axis=1)
    agedb30_std = np.expand_dims(np.asarray(agedb30_std, dtype=float),axis=1)

    mean_arr = np.concatenate((lfw_mean, agedb30_mean), axis=1)
    std_arr = np.concatenate((lfw_std**2, agedb30_std**2), axis=1)

    mean_auc = np.mean(mean_arr,axis=1)
    std_auc = np.sqrt(np.sum(std_arr,axis=1)/2)
    mean_std_auc = [str(mean) + '+-' + str(std) for mean, std in zip(mean_auc, std_auc)]

    sorted_mean_auc = np.argsort(mean_auc)[::-1]
    order = np.zeros(len(sorted_mean_auc))
    for ii, i in enumerate(sorted_mean_auc):
        order[i] = int(ii + 1)

    df['averaged_auc'] = mean_std_auc
    df['best'] = order

def split_df(df, split):
    def make_dict(max_dict, values, col_idx):
        max_val = max(values)
        max_values_loc = [ i for i in range(len(values)) if values[i] == max_val ]
        max_dict[col_idx] = max_values_loc

    max_dict = {}
    selected_split_df = df.iloc[8*(split - 1) : 8*split]
    selected_split_df.sort_values(by='Model', key=lambda x: x.map(ORDER_MODEL), inplace=True)
    selected_split_df = selected_split_df.iloc[:5]
    columns = selected_split_df.columns.to_list()
    #pdb.set_trace()
    columns = [columns[1]] + [columns[0]] + columns[3:]  
    selected_split_df = selected_split_df[columns]
    add_columns(selected_split_df)
    for col_idx in range(2, 7):
        values = [round(float(i.split('+-')[0]), ROUNDING) for i in np.squeeze(selected_split_df.iloc[:, [col_idx]].to_numpy())]
        make_dict(max_dict, values, col_idx)
    return selected_split_df, max_dict

row_in_latex = ''
splitter = '+-'
splits_number = int(len(the_file) / 8)
the_file.sort_values(by=' Masked Dataset', key=lambda x: x.map(ORDER_DB), inplace=True)
for split in range(1, splits_number + 1):
    print(the_file.iloc[(split-1)*8][1])
    database_name = change_db_type_name(the_file.iloc[(split-1)*8][1])
    if database_name != NOT_TO_PROCESS:
       row_in_latex += f'\centering\multirow{{5}}{{*}}' + '{'+database_name+'}'      
       selected_split_df, max_dict = split_df(the_file, split)
       for row_i, (idx, row) in enumerate(selected_split_df.iterrows()):
           row_in_latex += ' & ' + change_images_type_name(row[1])
           row_in_latex = make_row(row_in_latex, row, max_dict, row_i)
           first_row_ind = False
           row_in_latex += r' \tabularnewline \cline{2-8} '
       row_in_latex += ' \hline '

print(row_in_latex)
