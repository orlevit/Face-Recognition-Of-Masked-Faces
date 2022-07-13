# used a threshold that learned from the validation set DURING TRAINING AND IN THE MODEL DICTONARY and then apply it to 10 fold split on the test set
import os
import sys
import time
import math
import torch
import random
import argparse
import numpy as np
from datetime  import datetime
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from create_emb import *
from ijbc_config import *
sys.path.append(os.path.realpath('../../../model_training'))
from models_architecture import *
from helper import get_optimizer, one_epoch_run


def parse_arguments():
    parser = argparse.ArgumentParser(description='do verification')
    parser.add_argument('-s', '--split', type=int, help='The split number')
    args = parser.parse_args()

    return args

class EmbbedingsDataset(Dataset):
    def __init__(self, selected_data, selected_labels):
        self.data = selected_data
        self.labels =  selected_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embeddings1 = np.concatenate(np.expand_dims([emb[2 * idx, :] for emb in self.data],axis=0), axis=1)
        embeddings2 = np.concatenate(np.expand_dims([emb[2 * idx + 1, :] for emb in self.data],axis=0), axis=1)
        label = self.labels[idx]
        return embeddings1, embeddings2, label
   
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class LFold:
    def __init__(self, n_splits=2, shuffle=False, random_state=42):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def load_tested_model():
    print(COMPOSED_MODEL_PATH)
    model = NeuralNetwork15()   # Make sure the mode architecture!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(COMPOSED_MODEL_PATH)
    model_state_dict = checkpoint["model_state_dict"]
    best_threshold = checkpoint["threshold"]
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model, best_threshold, device

def load_transfer_models():
    models_dir, epochs = set_models_epochs(BASE_TRANS_MODEL_DIR)
    models, _, models_names = set_models(models_dir, epochs, IMAGE_SIZE, batch_size=1)
    return models, models_names

def ijbc_data_and_labels(selected_tmps, transfer_models, ie_idx, ie_arr):
    for ii, tmp_id in enumerate(selected_tmps, 1):
        print(f' index:{ii}, templetae:{tmp_id}')
        _, _, ie_arr, ie_idx = templeate_to_embs(tmp_id, transfer_models, ie_idx, ie_arr)

    return ie_idx, ie_arr

def get_selected_templeates(args):
    uniq_tmp_ids = pd.unique(TEMPLATE_DF['TEMPLATE_ID'])
    indices = np.arange(len(uniq_tmp_ids))
    selected_length =  len(uniq_tmp_ids) // MAKE_EMB_SPLIT_NUM + 1
    selected_tmp = uniq_tmp_ids[(args.split - 1) * selected_length : args.split * selected_length]
    return selected_tmp

def main():
    transfer_models, transfer_models_names = load_transfer_models()
    print(f'Transfer models, names: {transfer_models_names}')
    classification_list = []
    ie_idx = []
    ie_arr = None

    tic_split = time.time() 
    selected_tmps = get_selected_templeates(args)
    print(f'Length of uniqes: {len(selected_tmps)}')  
    ie_idx, ie_arr =  ijbc_data_and_labels(selected_tmps, transfer_models, ie_idx, ie_arr)
    np.save(IMG_EMB_ARR.format(args.split),ie_arr)
    np.save(IMG_IDX_ARR.format(args.split),ie_idx)

    print(f' The current images proccessed number is: {len(ie_idx)}')
    split_time = time.time() - tic_split
    print(f'Total time:: {split_time}')


if __name__ == '__main__':
   args = parse_arguments()
   main()
