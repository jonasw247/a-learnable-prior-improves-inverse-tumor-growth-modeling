import time
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle5 as pickle
import os
from datetime import datetime
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary_string
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from network import *

def train(num_epochs, model, train_dataset, train_generator, optimizer, device, lossfunction, writer,
              numberofbatches, numberofbatches_val,step,
              reduce_lr_on_flat, scheduler, val_generator, step_val, best_val_loss, savelogdir, writerval,
              dropoutrate, batch_size, numoutputs, learning_rate, lr_scheduler_rate, starttrain, endtrain, startval,
              endval, version, schedulername, lossfunctionname, seed, is_debug, optimizername, weight_decay_sgd,
              modelfun_name, lr_patience, includesft, outputmode, args, summarystring, additionalsummary,
              is_finetune = False):

    lastsavetime = time.time()
    if is_finetune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    for epoch in range(num_epochs):
        #training
        train_dataset.epoch = epoch
        running_training_loss = 0.0
        model = model.train()
        for batch_idx, (x,y) in enumerate(train_generator):
            #x: volume
            #y: parameters
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            y_predicted = model(x)
            cost = lossfunction(y_predicted, y)
            #cost = F.mse_loss(y_predicted, y)
            cost.backward()

            optimizer.step()


            running_training_loss += cost
            writer.add_scalar('Loss/train', cost, step)

            myLoss = torch.mean((y_predicted - y)**2, axis = 0)

            if outputmode == 7:
                writer.add_scalar('Loss/train_mu1', myLoss[0], step)
                writer.add_scalar('Loss/train_mu2', myLoss[1], step)
            
            if outputmode == 8:
                writer.add_scalar('Loss/train_mu1', myLoss[0], step)
                writer.add_scalar('Loss/train_mu2', myLoss[1], step)
                writer.add_scalar('Loss/train_x', myLoss[2], step)
                writer.add_scalar('Loss/train_y', myLoss[3], step)
                writer.add_scalar('Loss/train_z', myLoss[4], step)

            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.6f'
                   %(epoch+1, num_epochs, batch_idx + 1,
                     numberofbatches, cost))

            step += 1

            if not reduce_lr_on_flat:
                scheduler.step()
                print(scheduler.get_last_lr())
                writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch * numberofbatches + batch_idx)

        total_train_loss = (running_training_loss / numberofbatches).item() #avg loss in epoch!
        print('Epoch: %03d | Average loss: %.6f'%(epoch+1, total_train_loss))
        writer.add_scalar('Loss/avg_train', total_train_loss, epoch)
        writer.add_scalar('losses', total_train_loss, epoch)

        #validation
        running_validation_loss = 0.0
        model = model.eval()
        with torch.set_grad_enabled(False):
            for batch_idy, (x,y) in enumerate(val_generator):
                x, y = x.to(device), y.to(device)
                y_predicted = model(x)
                cost = lossfunction(y_predicted, y)
                #cost = F.mse_loss(y_predicted, y)

                running_validation_loss += cost

                myLoss = torch.mean((y_predicted - y)**2, axis = 0)

                if outputmode == 7:
                    writer.add_scalar('Loss/val_mu1', myLoss[0], step)
                    writer.add_scalar('Loss/val_mu2', myLoss[1], step)
            
                if outputmode == 8:
                    writer.add_scalar('Loss/val_mu1', myLoss[0], step)
                    writer.add_scalar('Loss/val_mu2', myLoss[1], step)
                    writer.add_scalar('Loss/val_x', myLoss[2], step)
                    writer.add_scalar('Loss/val_y', myLoss[3], step)
                    writer.add_scalar('Loss/val_z', myLoss[4], step)
                
                writer.add_scalar('Loss/val', cost, step_val)
                print ('Validating: %03d | Batch %03d/%03d | Cost: %.4f'
                   %(epoch+1, batch_idy + 1, numberofbatches_val, cost))
                step_val += 1

        total_val_loss = (running_validation_loss / numberofbatches_val).item()
        print('Epoch: %03d | Validation loss: %.6f'%(epoch+1, total_val_loss))
        writer.add_scalar('Loss/avg_val', total_val_loss, epoch)
        writerval.add_scalar('losses', total_val_loss, epoch)

        if np.round(total_val_loss,8) < np.round(best_val_loss,8):
            best_val_loss = total_val_loss
            save_inverse_model(savelogdir, epoch, model.state_dict(), optimizer.state_dict(), best_val_loss,
                                total_train_loss, dropoutrate, batch_size, numoutputs, learning_rate,
                                lr_scheduler_rate, starttrain, endtrain, startval, endval, version,
                                schedulername, lossfunctionname, seed, is_debug,
                                optimizername, weight_decay_sgd, modelfun_name, lr_patience, includesft, outputmode,
                                str(args),
                                summarystring, additionalsummary, '/bestval-model.pt')
            print(">>> Saving new model with new best val loss " + str(best_val_loss))

        if (time.time() - lastsavetime) > 3600.0: #more than an hour has passed since last save
            save_inverse_model(savelogdir, epoch, model.state_dict(), optimizer.state_dict(), best_val_loss,
                               total_train_loss, dropoutrate, batch_size, numoutputs, learning_rate,
                               lr_scheduler_rate, starttrain, endtrain, startval, endval, version,
                               schedulername, lossfunctionname, seed, is_debug,
                               optimizername, weight_decay_sgd, modelfun_name, lr_patience, includesft, outputmode,
                               str(args),
                               summarystring, additionalsummary, '/epoch' + str(epoch) + '.pt')
            lastsavetime = time.time()
            print(">>> Saved!")

        if reduce_lr_on_flat:
            scheduler.step(total_val_loss)
            print(optimizer.param_groups[0]['lr'])
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)


    print("Best val loss: %.6f"%(best_val_loss))
    writer.close()
    writerval.close()


def fine_tune_last_layer(num_epochs, model, train_generator, val_generator, lossfunction, optimizer, device, savelogdir, writer, args):
    """
    Fine-tunes the last layer of a given model.
    
    Parameters:
        num_epochs (int): Number of training epochs.
        model (torch.nn.Module): Pre-trained model with the last layer to be fine-tuned.
        train_generator (DataLoader): DataLoader for the training dataset.
        val_generator (DataLoader): DataLoader for the validation dataset.
        lossfunction (torch.nn.Module): Loss function to be used during training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (str): Device to run the model on ('cpu' or 'cuda').
        savelogdir (str): Directory path to save the fine-tuned model.
        writer: TensorBoard writer for logging.
        args: Additional arguments (if any) needed for training.

    Returns:
        float: Best validation loss achieved during fine-tuning.
    """

    # Set model to the device
    model.to(device)

    best_val_loss = float('inf')
    lastsavetime = time.time()

    # Freeze all layers except the last layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    step = 0
    step_val = 0

    for epoch in range(num_epochs):
        print('starting epoch ' + str(epoch + 1))
        # Training
        model.train()
        running_training_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_generator):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_predicted = model(x)
            cost = lossfunction(y_predicted, y)
            cost.backward()
            optimizer.step()

            running_training_loss += cost.item()
            writer.add_scalar('Loss/train', cost.item(), step)
            step += 1

        total_train_loss = running_training_loss / len(train_generator)
        print('Epoch: %03d | Average training loss: %.6f' % (epoch + 1, total_train_loss))
        writer.add_scalar('Loss/avg_train', total_train_loss, epoch)
        writer.add_scalar('losses', total_train_loss, epoch)

        # Validation
        model.eval()
        running_validation_loss = 0.0
        with torch.set_grad_enabled(False):
            for batch_idy, (x, y) in enumerate(val_generator):
                x, y = x.to(device), y.to(device)
                y_predicted = model(x)
                cost = lossfunction(y_predicted, y)
                running_validation_loss += cost.item()
                writer.add_scalar('Loss/val', cost.item(), step_val)
                step_val += 1

        total_val_loss = running_validation_loss / len(val_generator)
        print('Epoch: %03d | Validation loss: %.6f' % (epoch + 1, total_val_loss))
        writer.add_scalar('Loss/avg_val', total_val_loss, epoch)
        writer.add_scalar('losses', total_val_loss, epoch)

        # Save the model if the validation loss improves
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), savelogdir + '/bestval-model.pt')
            print(">>> Saving new model with new best val loss " + str(best_val_loss))

        # Save the model periodically (e.g., every hour)
        if (time.time() - lastsavetime) > 3600.0:
            torch.save(model.state_dict(), savelogdir + '/epoch' + str(epoch) + '.pt')
            lastsavetime = time.time()
            print(">>> Saved!")

    print("Best validation loss: %.6f" % (best_val_loss))
    writer.close()
    return best_val_loss
