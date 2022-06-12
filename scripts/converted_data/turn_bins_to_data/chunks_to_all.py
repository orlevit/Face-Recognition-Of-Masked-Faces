# joined the chuncked data into one all file(e.g. chunks of 2k images in 700k images will joined all  70k/2k=35 files into one file)
import os
import torch
import argparse
import mxnet as mx
import numpy as np
from pprint import pprint
from datetime import datetime

#DATA_PATH = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000'
#TARGET_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000/all'
MASK = 'covid19'
#DATA_PATH = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/tmp/test/db_nomask_lfw'
#TARGET_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/ready_data/350000_test_lfw_casia_pairs/nomask'

def return_input_loc(args):
    DATA_PATH = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/a{args.mask}mask/train/db_a{args.mask}mask_a{args.mask}mask'
    TARGET_LOC = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/a{args.mask}mask/train/all_350k_pairs'
    return DATA_PATH, TARGET_LOC

def parse_arguments():
    parser = argparse.ArgumentParser(description='do verification')
    parser.add_argument('-m', '--mask', type=str, help='The mask to joined all the small chuncks of it to one file that joined them all')
    args = parser.parse_args()

    return args

def get_files_path(data_path): 
    print(data_path)
    data_path_list = []
    labels_path_list = []
    
    for cur_dir1, dirs1, _ in os.walk(data_path):
        for sub_dir in dirs1:
            for cur_dir2, _, files in os.walk(os.path.join(cur_dir1, sub_dir)):
                for file in files:
                    path = os.path.join(cur_dir2,file)
                    name_of_file = path.rsplit('/',1)[-1] 
                    
                    if name_of_file.startswith('data'): 
                        data_path_list.append(path)
                    if name_of_file.startswith('labels'): 
                        labels_path_list.append(path)
                   
    data_path_list = sorted(data_path_list, key=lambda path:int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    labels_path_list = sorted(labels_path_list, key=lambda path: int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    pprint(data_path_list)
    pprint(labels_path_list)
    return data_path_list, labels_path_list

def compose_data(paths, is_data):
    all_data = None
    for path in paths:
        print('Processing file: ', path)
        tic = datetime.now()
        if is_data:
           loaded_numpy = np.expand_dims(mx.nd.load(path)[0].asnumpy(), axis=0) # load mxnet/torch or numpy
        else:
           loaded_numpy = np.expand_dims(mx.nd.load(path)[0].asnumpy(), axis=0) # load mxnet/torch or numpy
        if all_data is None:
            all_data = loaded_numpy
        else:
            all_data = np.concatenate((all_data, loaded_numpy), axis=1) # to contatenate according o an axis - change to poper use
            
        del loaded_numpy
        toc = datetime.now()
        print('Time: ',toc-tic)
        print(all_data.shape)
    return all_data 

def main(args):
    data_path, target_loc = return_input_loc(args)
    data_path_list, labels_path_list =  get_files_path(data_path)
    data = compose_data(data_path_list, True)
    lables = compose_data(labels_path_list, False)
    
    os.makedirs(target_loc)
    data_target_loc = os.path.join(target_loc, 'data.npy')
    labels_target_loc = os.path.join(target_loc, 'labels.npy')
    torch.save(data, data_target_loc, pickle_protocol=4)
    torch.save(lables, labels_target_loc, pickle_protocol=4)

if __name__ == '__main__':
   args = parse_arguments()
   main(args)
