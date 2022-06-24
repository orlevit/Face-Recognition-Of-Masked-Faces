# used a threshold that learned from the validation set DURING TRAINING AND IN THE MODEL DICTONARY and then apply it to 10 fold split on the test set
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

BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV2'
#BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/images_masked_crop/sample_ROF/wsnp/covid19/combined/db_for_test_combinedV2'
#BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV1'
BASE_MODELS_PATH = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/models'

# Combined version1
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000_pairs_batch_all_NeuralNetwork5_lr(1e-07,)_32_D19_02_2022_T22_06_17_833096.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000_pairs_batch_all_hidden4096_NeuralNetwork5_lr1e-05_32_D20_02_2022_T18_53_58_770221.pt')
# Combined version2
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '300000_pairs_same_masks_hidden4096_NeuralNetwork5_lr1e-07_32_D30_05_2022_T16_54_35_404599.pt')

### Retrain
# Version1
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-05_32_D17_06_2022_T16_43_43_708447.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-06_32_D17_06_2022_T16_41_52_082341.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-07_32_D17_06_2022_T16_40_49_981712.pt')
# Version2
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-05_32_D17_06_2022_T16_44_18_742560.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-06_32_D17_06_2022_T16_42_13_910480.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-07_32_D17_06_2022_T16_41_08_309996.pt')
MODEL_PATH = os.path.join(BASE_MODELS_PATH, '20000pairsV2_long_reduce2_seed42_NeuralNetwork14_lastHidden4096_lr1e-07_32_D24_06_2022_T01_04_00_719484.pt')
MODEL_PATH = os.path.join(BASE_MODELS_PATH, '20000pairsV2_long_reduce4_seed42_NeuralNetwork14_lastHidden4096_lr1e-07_32_D24_06_2022_T01_13_29_644254.pt')

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
    print(data.shape,mask.shape)
    #import pdb;pdb.set_trace();
    selected_data = data[:,np.repeat(mask, 2),:]
    testDataset = EmbbedingsDataset(selected_data, selected_labels, TEST_DS_IND)
    test_dataloader = DataLoader(testDataset, batch_size=len(testDataset), shuffle=False)

    return test_dataloader

def main():
    print(MODEL_PATH)
    model = NeuralNetwork14()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH)
    model_state_dict = checkpoint["model_state_dict"]
    best_threshold = checkpoint["threshold"]
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()
    optimizer, _ = get_optimizer(1, model)
    k_fold = LFold(n_splits=10, shuffle=False, random_state=None)

    for cur_dir, subFolders, _ in os.walk(BASE_DATA_LOC):
        for sub_dir in subFolders:
            data = np.load(os.path.join(cur_dir, sub_dir, 'data.npy'))
            labels = list(np.load(os.path.join(cur_dir, sub_dir, 'labels.npy'))[0,:])
            print(f'data shape: {data.shape}, labels length: {len(labels)}, threshold: {best_threshold}')
            indices = np.arange(len(labels))
            classification_list = []

            blockPrint()
            for _, test_indices in k_fold.split(indices):
                #import pdb;pdb.set_trace();
                test_dataloader = get_test_data_by_indices(data, labels, test_indices)
                avg_loss, avg_classificatin_loss, _, run_time = one_epoch_run(test_dataloader, optimizer, model, loss_fn, device, train_ind=False, best_threshold=best_threshold)
                classification_list.append(avg_classificatin_loss)
                print(f'Train: run_time:{run_time=}, loss:{avg_loss=} classification accuracy:{avg_classificatin_loss=}')
            enablePrint()
            print(f'Images:{sub_dir}: {np.mean(classification_list)}+-{np.std(classification_list)}')
            print('---------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()

