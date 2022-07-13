# find a threshold on the 1/10 test set and then apply it to fold of the 9/10 test (10 cv) - this is like insightface did
import os
import sys
import math
import torch
import random
import numpy as np
from time import time
from datetime  import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
sys.path.append(os.path.realpath('../../model_training'))
from config import SPLIT_TRAIN, TRAIN_DS_IND, VALID_DS_IND, WHOLE_DATA_BATCH, TEST_DS_IND
from models_architecture import *
from helper import EmbbedingsDataset, get_optimizer, select_train_valid, find_a_threshold, calculate_accuracy

BASE_MODELS_PATH = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/models'

#TRAIN_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/combinedV1_multi_masks_350000/all'
#TRAIN_DATA_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'data.pt')
#TRAIN_LABELS_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'labels.pt')

# Combined version2
#TRAIN_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/all'
#TRAIN_DATA_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'data.npy')
#TRAIN_LABELS_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'labels.npy')
#TEST_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV2'
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '300000_pairs_same_masks_hidden4096_NeuralNetwork5_lr1e-07_32_D30_05_2022_T16_54_35_404599.pt')

# Data
# Combined version1
#TEST_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV1'
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000_pairs_batch_all_hidden4096_NeuralNetwork5_lr1e-05_32_D20_02_2022_T18_53_58_770221.pt')

# Combined version2
#TEST_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV2'
TEST_BASE_DATA_LOC = '/RG/rg-tal/orlev/datasets/original_ds/MFR2/converted_data/db_for_test_combinedV2'

### Retrain
# Version1
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-05_32_D17_06_2022_T16_43_43_708447.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-06_32_D17_06_2022_T16_41_52_082341.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-07_32_D17_06_2022_T16_40_49_981712.pt')
# Version2
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-05_32_D17_06_2022_T16_44_18_742560.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-06_32_D17_06_2022_T16_42_13_910480.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-07_32_D17_06_2022_T16_41_08_309996.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '20000pairsV2_long_reduce2_seed42_NeuralNetwork14_lastHidden4096_lr1e-07_32_D24_06_2022_T01_04_00_719484.pt')
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


def create_dataloaders_valid(train_data_loc, train_labels_loc, split_train, valid_ds_ind, batch_size):
    train_data = torch.load(train_data_loc)
    train_labels = torch.load(train_labels_loc)
    print(f'Train data shape: {train_data.shape}, train labels shape: {train_labels.shape}')

    # Numbers for split the data properly 
    mask =select_train_valid(train_data, split_train)

    validDataset = EmbbedingsDataset(train_data, train_labels, valid_ds_ind, mask)

    if batch_size == WHOLE_DATA_BATCH:
       batch_size_valid = len(validDataset)
    else:
       batch_size_valid = batch_size

    valid_dataloader = DataLoader(validDataset, batch_size=batch_size_valid, shuffle=False)

    return valid_dataloader

def creae_dataloaders_test(test_data_loc, test_labels_loc, split_train, batch_size, test_ds_ind):
    test_data = torch.load(test_data_loc)
    test_labels = torch.load(test_labels_loc)
    print(f'Test data shape: {test_data.shape}, test labels shape: {test_labels,shape}')
  
    testDataset = EmbbedingsDataset(test_data, test_labels, test_ds_ind)

    if batch_size == WHOLE_DATA_BATCH:
       batch_size_test = len(testDataset)
    else:
       batch_size_test = batch_size

    test_dataloader = DataLoader(testDataset, batch_size=batch_size_test, shuffle=False)

    return test_dataloader 

def one_epoch_run_threshold(train_dataloader, optimizer, model, loss_fn, device, threshold, train_ind=False):
    last_loss = 0.
    running_loss = 0.
    running_classificatin_loss = 0.
    tic = datetime.now()

    model.train(train_ind)
 
    for i, data in enumerate(train_dataloader):
        emb1, emb2, labels = data
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(emb1.float(), emb2.float())
        converted_labels = labels.type(torch.float)[:, None]
        converted_labels[converted_labels == 0] = -1
        loss = loss_fn(outputs, converted_labels.float())

        max_threshold, max_accuracy = find_a_threshold(outputs, labels, train_ind, best_threshold=threshold)
        running_classificatin_loss += max_accuracy
        
    run_time = round((datetime.now() - tic).total_seconds(), 1)
    avg_classificatin_loss = running_classificatin_loss / len(train_dataloader)
    avg_loss = running_loss / len(train_dataloader)

    return avg_loss, avg_classificatin_loss, run_time

def find_best_threshold(valid_dataloader, model, device, fold_num):
    thresholds = np.arange(-0.1, 0.1, 0.005)
    nrof_thresholds = len(thresholds)
    tprs = np.zeros(nrof_thresholds)
    fprs = np.zeros(nrof_thresholds)
    accuracy = np.zeros(nrof_thresholds)

    for i, data in enumerate(valid_dataloader):
        emb1, emb2, labels = data
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 
        outputs = model(emb1.float(), emb2.float())
        max_threshold, max_accuracy = find_a_threshold(outputs, labels, train_ind=True, best_threshold=None)
        print(f" THe best accuracy on the {fold_num}/(10) fold train set is:{max_accuracy} and the threshold is:{max_threshold}")
    return max_threshold

###########################################################################
#def calculate_accuracy(threshold, dist, actual_issame):
#    predict_issame = torch.gt(dist, threshold)
#    tp = torch.sum(torch.logical_and(predict_issame, actual_issame))
#    fp = torch.sum(torch.logical_and(predict_issame, torch.logical_not(actual_issame)))
#    tn = torch.sum(
#        torch.logical_and(torch.logical_not(predict_issame),
#                       torch.logical_not(actual_issame)))
#    fn = torch.sum(torch.logical_and(torch.logical_not(predict_issame), actual_issame))
#
#    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
#    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
#    acc = float(tp + tn) / len(dist)
#    return tpr, fpr, acc

#def calculate_accuracy2(threshold, dist, actual_issame):
#    predict_issame = np.less(dist, threshold)
#    tp = np.sum(np.logical_and(predict_issame, actual_issame))
#    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#    tn = np.sum(
#        np.logical_and(np.logical_not(predict_issame),
#                       np.logical_not(actual_issame)))
#    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
#
#    import pdb;pdb.set_trace();
#    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
#    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
#    acc = float(tp + tn) / dist.size
#    return tpr, fpr, acc

def get_test_data_by_indices(data, labels, indices):
    mask = np.zeros(len(labels), dtype=bool)
    mask[indices] = True
    selected_labels = list(np.array(labels)[mask])
    print(data.shape,mask.shape, sum(selected_labels))
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
    #model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()
    optimizer, _ = get_optimizer(1, model)
    k_fold = LFold(n_splits=10, shuffle=False, random_state=None)

    #valid_dataloader = create_dataloaders_valid(TRAIN_DATA_LOC, TRAIN_LABELS_LOC, SPLIT_TRAIN, VALID_DS_IND, WHOLE_DATA_BATCH) 
    #threshold = find_best_threshold(valid_dataloader, model, device)

    for cur_dir, subFolders, _ in os.walk(TEST_BASE_DATA_LOC):
        for sub_dir in subFolders:
            test_data = np.load(os.path.join(cur_dir, sub_dir, 'data.npy'))
            test_labels = list(np.load(os.path.join(cur_dir, sub_dir, 'labels.npy'))[0,:])
            print(f'Data loc:{os.path.join(cur_dir, sub_dir)}, data shape: {test_data.shape}, labels length: {len(test_labels)}')
            indices = np.arange(len(test_labels))
            classification_list = []
            thresholds_list = []

            blockPrint()
            for fold_num,(train_indices, test_indices) in enumerate(k_fold.split(indices)):
                train_dataloader = get_test_data_by_indices(test_data, test_labels, train_indices)
                threshold = find_best_threshold(train_dataloader, model, device, fold_num)
                test_dataloader = get_test_data_by_indices(test_data, test_labels, test_indices)
                print(f'Split number:{fold_num}\nTrain data set size(90%): {len(train_dataloader.dataset)} Test data set size(10%): {len(test_dataloader.dataset)}')
                avg_loss, avg_classificatin_loss, run_time = one_epoch_run_threshold(test_dataloader, optimizer, model, loss_fn, device, threshold, train_ind=False)
                classification_list.append(avg_classificatin_loss)
                print(f'Test: run_time:{run_time}, loss={avg_loss}, classification accuracy:{avg_classificatin_loss}, thresold:{threshold}')
                thresholds_list.append(threshold)
            enablePrint()
            print(f'Images:{sub_dir}: {np.mean(classification_list)}+-{np.std(classification_list)}, avg_thresold={np.mean(thresholds_list)} ,{thresholds_list=}')
            print('---------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()

