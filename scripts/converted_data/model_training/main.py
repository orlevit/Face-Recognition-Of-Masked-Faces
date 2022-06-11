import os
import torch
from datetime  import datetime
from models_architecture import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import TRAIN_DATA_LOC, TRAIN_LABELS_LOC, TEST_DATA_LOC, TEST_LABELS_LOC, SPLIT_TRAIN, BATCH_SIZE, RUN_DIR, EMBBEDINGS_NUMBER, MODELS_NUMBER, \
                   TRAIN_DS_IND, VALID_DS_IND, TEST_DS_IND, EPOCHS, MODELS_SAVE_PATH, MIN_LOSS_SAVE, EARLY_STOP_DIFF, EMBBEDINGS_REDUCED, LINEAR_INIT, BILINEAR_INIT, WHOLE_DATA_BATCH
from helper import one_epoch_run, create_dataloaders, get_optimizer, parse_arguments, initialize_weights 

def main(args):
     train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(TRAIN_DATA_LOC, TRAIN_LABELS_LOC, TEST_DATA_LOC, TEST_LABELS_LOC, \
                                                                              SPLIT_TRAIN, TRAIN_DS_IND, VALID_DS_IND, WHOLE_DATA_BATCH, TEST_DS_IND)
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print(f'The model will run on device: {device}')
     print(f'Train samples:{len(train_dataloader.dataset)}, Valid samples:{len(valid_dataloader.dataset)}, Test samples:{len(test_dataloader.dataset)}')
     # set model parameters
     model = NeuralNetwork5()
     model.apply(initialize_weights(LINEAR_INIT, BILINEAR_INIT))
     model.to(device)
     loss_fn = torch.nn.MSELoss()
     #loss_fn = torch.nn.CrossEntropyLoss()#L1Loss()
     optimizer, opt_params = get_optimizer(args.combination_number, model)

     # Set training logs
     model_name = f'300000_pairs_same_masks_hidden4096_{type(model).__name__}_lr{opt_params[0]}_{EMBBEDINGS_REDUCED}_{datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S_%f")}'
     print(model_name)
     writer = SummaryWriter(os.path.join(RUN_DIR, model_name))
     
     # Set training parameters 
     epoch = 1
     last_updated_epoch = 1
     continue_training = True
     best_cvloss = -float('inf')
     
     while epoch < EPOCHS and continue_training:
         avg_loss, avg_ctrain_loss, ttime = one_epoch_run(train_dataloader, optimizer, model, loss_fn, device, train_ind=True)
         # We don't need gradients on to do reporting
         avg_vloss, avg_cvalid_loss, vtime = one_epoch_run(valid_dataloader, optimizer, model, loss_fn, device, train_ind=False)
     
         print('Epoch:{}/{}. Time- Train:{} Valid:{}. Loss- Train:{} Valid:{}, classification acc:- Train:{} Valid:{}'.format(
                epoch, EPOCHS, ttime, vtime, avg_loss, avg_vloss, avg_ctrain_loss, avg_cvalid_loss))

         # Lof to tensorboard
         writer.add_scalars('Loss', {'Train': avg_loss, 'Valid': avg_vloss}, epoch)
         writer.add_scalars('Accuracy', {'Train': avg_ctrain_loss, 'Valid': avg_cvalid_loss}, epoch)
     
         # Track best performance, and save the model's state or do early stopping
         if best_cvloss < avg_cvalid_loss: 
             best_cvloss = avg_cvalid_loss
             last_updated_epoch = epoch
     
             if  avg_vloss < MIN_LOSS_SAVE:
                 model_path = os.path.join(MODELS_SAVE_PATH, model_name+'.pt') 
                 torch.save(model.state_dict(), model_path)
                 print(f'Model saved, epoch number:{epoch}, Loss- Train:{round(avg_loss)} Valid:{round(avg_vloss)}')
         if EARLY_STOP_DIFF < epoch - last_updated_epoch:
            continue_training = False
     
         epoch += 1
     
     avg_test_loss, avg_ctest_loss, test_time = one_epoch_run(test_dataloader, optimizer, model, loss_fn, device, train_ind=False)
     print('\n\n\n\n\n')
     print('--------------------------   TEST   --------------------------')
     print('Epoch:{}. Time- Test:{}. Loss- Test:{}, classification acc:- Test:{}'.format(epoch, test_time, round(avg_test_loss,5), avg_ctest_loss))

if __name__ == '__main__':
   args = parse_arguments()
   main(args)

