# used a threshold that learned from the validation set DURING TRAINING AND IN THE MODEL DICTONARY and then apply it to 10 fold split on the test set
import os
import sys
import math
import time
import torch
import random
import numpy as np
from datetime  import datetime
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

from create_emb import *
from ijbc_config import *
sys.path.append(os.path.realpath('../../../model_training'))
from models_architecture import *
from helper import get_optimizer, one_epoch_run


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

def get_test_data_by_indices(indices, transfer_models, ie_idx, ie_arr):
    selected_data, selected_labels, ie_idx, ie_arr =  ijbc_data_and_labels(indices, transfer_models, ie_idx, ie_arr)
    testDataset = EmbbedingsDataset(selected_data, selected_labels)
    test_dataloader = DataLoader(testDataset, batch_size=len(testDataset), shuffle=False)

    return test_dataloader, ie_idx, ie_arr

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

def load_data_arraies():
    if os.path.exists(IMG_EMB_ARR_ALL) and os.path.exists(IMG_IDX_ARR_ALL):
       ie_idx = list(np.load(IMG_IDX_ARR_ALL))
       ie_arr = np.load(IMG_EMB_ARR_ALL)
    else:
       ie_idx = []        
       ie_arr = None
    return ie_idx, ie_arr

def main():
    print('The templetae file', IJBC_TEMPLATE_LOC)
    model, best_threshold, device = load_tested_model()
    loss_fn = torch.nn.MSELoss()
    optimizer, _ = get_optimizer(1, model)
    k_fold = LFold(n_splits=SPLIT_NUM, shuffle=False, random_state=None)
    transfer_models, transfer_models_names = load_transfer_models()
    print(f'Transfer models, names: {transfer_models_names}')
    indices = np.arange(len(MATCH_DF))
    classification_list = []
    ie_idx, ie_arr = load_data_arraies()

    #blockPrint()
    for split_num, (_, test_indices) in enumerate(k_fold.split(indices), 1):
        tic_split = time.time()
        print(f'Split number: {split_num}')  
        test_dataloader, ie_idx, ie_arr = get_test_data_by_indices(test_indices, transfer_models, ie_idx, ie_arr)
        avg_loss, avg_classificatin_loss, _, run_time = one_epoch_run(test_dataloader, optimizer, model, loss_fn, device, train_ind=False, best_threshold=best_threshold)
        classification_list.append(avg_classificatin_loss)
        split_time = time.time() - tic_split
        print(f'Time: split_time: {split_time}, run_time:{run_time}, loss:{avg_loss} classification accuracy:{avg_classificatin_loss}')
    #enablePrint()
    print(f'Images:{sub_dir}: {np.mean(classification_list)}+-{np.std(classification_list)}')
    print('---------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
