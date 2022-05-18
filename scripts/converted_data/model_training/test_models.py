import os
import numpy as np
from glob import iglob
from datetime import datetime
from models_architecture import *
from torch.utils.data import Dataset, DataLoader
from helper import one_epoch_run, EmbbedingsDataset
from config import WHOLE_DATA_BATCH, TEST_DS_IND

MODELS_DIR = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/models'
TEST_DATA_BASE_DIR = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/tmp/test'
TEST_DATA_LOC = os.path.join(TEST_DATA_BASE_DIR, 'db_covid19mask_lfw/covid19_model/data.npy')
TEST_LABELS_LOC = os.path.join(TEST_DATA_BASE_DIR, 'db_covid19mask_lfw/covid19_model/labels.npy')


def create_test_dataloaders(test_data_loc, test_labels_loc, batch_size, test_ds_ind):
    test_data = np.load(test_data_loc)
    test_labels = np.load(test_labels_loc)
    #test_data = torch.load(test_data_loc)
    #test_labels = torch.load(test_labels_loc)
  
    tic = datetime.now()
    testDataset = EmbbedingsDataset(test_data, test_labels, test_ds_ind)

    if batch_size == WHOLE_DATA_BATCH:
       batch_size_test = len(testDataset)
    else:
       batch_size_test = batch_size

    test_dataloader = DataLoader(testDataset, batch_size=batch_size_test, shuffle=False)
    toc = datetime.now()
    print(f'Finish loading models: {toc-tic}')
    return test_dataloader 


test_dataloader = create_test_dataloaders(TEST_DATA_LOC, TEST_LABELS_LOC, WHOLE_DATA_BATCH, TEST_DS_IND)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The model will run on device: {device}')
print(f'Test samples:{len(test_dataloader.dataset)}, number of batches: {len(test_dataloader)}')

# set model parameters
# ------------------------------------ the wanted model ----------------------------------------
MODEL_STATE_DICT_LOC = os.path.join(MODELS_DIR, 'NeuralNetwork5_1_32_D16_02_2022_T18_36_54_292575.pt')
model = NeuralNetwork5()
model.load_state_dict(torch.load(MODEL_STATE_DICT_LOC))
# ----------------------------------------------------------------------------------------------
model.to(device)
model.eval()    
last_loss = 0.
running_loss = 0.
tic = datetime.now()

for i, data in enumerate(test_dataloader):
    emb1, emb2, labels = data
    emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 
    outputs = model(emb1.float(), emb2.float())
    converted_labels = labels.type(torch.float)[:, None]
    converted_labels[converted_labels == 0] = -1
    loss = loss_fn(outputs, converted_labels)
    running_loss += loss.item()
    
run_time = round((datetime.now() - tic).total_seconds(), 1)
classificatin_loss = np.round(sum((converted_labels.cpu().detach().numpy() * outputs.cpu().detach().numpy()) > 0) / len(converted_labels), 5)[0]
last_loss = running_loss / len(train_dataloader)

print('Test: Time:{}. classification accuracy:{}'.format(run_time, last_loss, classificatin_loss))
