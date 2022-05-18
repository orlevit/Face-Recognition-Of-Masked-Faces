# This tests only one images type(the covid19), in the 77,000  images pairs
import os
import sys
import math
import torch
import random
import numpy as np
from datetime  import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
sys.path.append(os.path.realpath('../model_training'))
from config import TEST_DS_IND
from models_architecture import *
from helper import EmbbedingsDataset, get_optimizer, one_epoch_run

BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/ready_data/350000_test_lfw_casia_pairs'
BASE_MODELS_PATH='/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/models'

MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000_pairs_batch_all_hidden4096_NeuralNetwork5_lr1e-05_32_D20_02_2022_T18_53_58_770221.pt')
TEST_DATA_LOC = os.path.join(BASE_DATA_LOC, 'data.pt')
TEST_LABELS_LOC = os.path.join(BASE_DATA_LOC, 'labels.pt')

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

def get_test_data_by_indices(data, labels, indices):
    mask = np.zeros(len(labels), dtype=bool)
    mask[indices] = True
    selected_labels = list(np.array(labels)[mask])
    selected_data = data[:,np.repeat(mask, 2),:]
    testDataset = EmbbedingsDataset(selected_data, selected_labels, TEST_DS_IND)
    test_dataloader = DataLoader(testDataset, batch_size=len(testDataset), shuffle=False)

    return test_dataloader

def main():

    model = NeuralNetwork5()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()
    optimizer, _ = get_optimizer(1, model)
    k_fold = LFold(n_splits=10, shuffle=False, random_state=None)

    for cur_dir, subFolders, _ in os.walk(BASE_DATA_LOC):
        for sub_dir in subFolders:
            data = torch.load(os.path.join(cur_dir, sub_dir, 'data.pt'))
            labels = list(torch.load(os.path.join(cur_dir, sub_dir, 'labels.pt'))[0,:])
            import pdb;pdb.set_trace()
            indices = np.arange(len(labels))
            classification_list = []

            blockPrint()
            for _, test_indices in k_fold.split(indices):
                #import pdb;pdb.set_trace();
                test_dataloader = get_test_data_by_indices(data, labels, test_indices)
                avg_loss, avg_classificatin_loss, run_time = one_epoch_run(test_dataloader, optimizer, model, loss_fn, device, train_ind=False)
                classification_list.append(avg_classificatin_loss)
                print(f'Train: run_time:{run_time=}, loss:{avg_loss=} classification accuracy:{avg_classificatin_loss=}')
            enablePrint()
            print(f'Images:{sub_dir}: {np.mean(classification_list)}+-{np.std(classification_list)}')
            print('---------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()

