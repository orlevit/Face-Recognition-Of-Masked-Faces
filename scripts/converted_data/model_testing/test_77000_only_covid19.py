# This tests only one images type(the covid19), in the 77,000  images pairs
import os
import sys
import torch
import math
import random
import numpy as np
from datetime  import datetime
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
sys.path.insert(0,'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_training')
from config import TRAIN_DATA_LOC, TRAIN_LABELS_LOC, TEST_DATA_LOC, TEST_LABELS_LOC, SPLIT_TRAIN, BATCH_SIZE, RUN_DIR, EMBBEDINGS_NUMBER, MODELS_NUMBER, \
                   TRAIN_DS_IND, VALID_DS_IND, TEST_DS_IND, EPOCHS, MODELS_SAVE_PATH, MIN_LOSS_SAVE, EARLY_STOP_DIFF, EMBBEDINGS_REDUCED, LINEAR_INIT, BILINEAR_INIT
from helper import one_epoch_run, create_dataloaders, get_optimizer, parse_arguments, initialize_weights , select_train_valid
from models_architecture import NeuralNetwork7

IMAGES_TYPE = 3
print(f'{IMAGES_TYPE=}')
def select_train_valid(train_data, split_train):
    models_num, imgs_num, embs_num = train_data.shape
    total_pairs_mask_img_num = (imgs_num // 2) / models_num
    train_mask_imgs_num = math.ceil((total_pairs_mask_img_num) * split_train / 2.0) * 2
    valid_mask_imgs_num = int(total_pairs_mask_img_num - train_mask_imgs_num)
    mask_data_chunck = np.r_[np.ones(train_mask_imgs_num), np.zeros(valid_mask_imgs_num)].astype(bool)
    random.seed(1)
    random.shuffle(mask_data_chunck)
    mask = np.tile(mask_data_chunck, models_num)
    return mask

def main(args):
     #train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(TRAIN_DATA_LOC, TRAIN_LABELS_LOC, TEST_DATA_LOC, TEST_LABELS_LOC, \
                                                                              #SPLIT_TRAIN, TRAIN_DS_IND, VALID_DS_IND, int(11000*0.8), TEST_DS_IND, args)
     data = torch.load(TRAIN_DATA_LOC)
     labels = torch.load(TRAIN_LABELS_LOC)
     test_data = torch.load(TEST_DATA_LOC)[IMAGES_TYPE,:,:]
     test_labels = torch.load(TEST_LABELS_LOC)
     mask = select_train_valid(data, SPLIT_TRAIN)

     mask = mask[:11000]
     data = data[:,:22000,:]

     train_data = data[IMAGES_TYPE, np.repeat(mask, 2), :]
     train_labels = labels[:len(mask)][mask]
     valid_data = data[IMAGES_TYPE, ~np.repeat(mask, 2), :]
     valid_labels = labels[:len(mask)][~mask]

     best_acc = 0
     best_threshold = 0
     thresholds = np.arange(0, 4, 0.01)

     norm1 = normalize(train_data[0::2])
     norm2 = normalize(train_data[1::2])
     diff = np.subtract(norm1, norm2)
     dist = np.sum(np.square(diff), 1)

     for threshold_idx, threshold in enumerate(thresholds):
         predict_issame = np.less(dist, threshold)
     #    import pdb;pdb.set_trace();
         tp = np.sum(np.logical_and(predict_issame, train_labels))
         tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(train_labels)))
         acc = float(tp + tn) / dist.size
         if best_acc < acc:
            best_acc = acc
            best_threshold = threshold
     print(f'Train: {best_acc=}, {best_threshold=}, True labels: {sum(train_labels)}/{len(train_labels)}, Max: {max(dist)}, Min: {min(dist)}')
       
     print('\n\n\n\n\n')
     print('--------------------------   TEST   --------------------------')
     norm1 = normalize(valid_data[0::2])
     norm2 = normalize(valid_data[1::2])
     diff = np.subtract(norm1, norm2)
     dist = np.sum(np.square(diff), 1)
     predict_issame = np.less(dist, best_threshold)
     tp = np.sum(np.logical_and(predict_issame, valid_labels))
     tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(valid_labels)))
     acc = float(tp + tn) / dist.size
     print(f'Valid: {acc=}, True labels: {sum(valid_labels)}/{len(valid_labels)}, Max: {max(dist)}, Min: {min(dist)}')
     norm1 = normalize(test_data[0::2])
     norm2 = normalize(test_data[1::2])
     diff = np.subtract(norm1, norm2)
     dist = np.sum(np.square(diff), 1)
     predict_issame = np.less(dist, best_threshold)
     tp = np.sum(np.logical_and(predict_issame, test_labels))
     tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(test_labels)))
     acc = float(tp + tn) / dist.size
     print(f'Test: {acc=}, True labels: {sum(test_labels)}/{len(test_labels)}, Max: {max(dist)}, Min: {min(dist)}')

if __name__ == '__main__':
   args = parse_arguments()
   main(args)

