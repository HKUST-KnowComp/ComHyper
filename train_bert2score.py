import sys
import time
import os
import logging
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as Dataset
import torch.distributed as dist
import torch.multiprocessing as mp

import configparser
from model.models import *
from utils.data_helper_4bert import Dataset

logger = logging.getLogger()

def make_hparam_string(config):
    hparam = "{}/s{}_h{}-{}_n{}_c{}-{}_b{}".format(
            config.get("hyperparameters", "model"),
            config.get("hyperparameters", "svd_dimension"),
            config.get("hyperparameters", "number_hidden_layers"),
            config.get("hyperparameters", "hidden_layer_size"),
            config.get("hyperparameters", "negative_num"),
            # config.get("hyperparameters", "weight_decay"),
            config.get("hyperparameters", "context_num"),
            config.get("hyperparameters", "context_len"),
            config.get("hyperparameters", "batch_size")
            )
    return hparam

def init_model(config, device):

    encoder_type = config.get("hyperparameters", "model")
    number_hidden_layers = int(config.getfloat("hyperparameters", "number_hidden_layers"))
    hidden_layer_size = int(config.getfloat("hyperparameters", "hidden_layer_size"))
    bert_dir = config.get("data", "bert_path")
    model = Bert2Score(encoder_type, bert_dir, hidden_layer_size, 0.1)
    #torch.distributed.init_process_group(backend="nccl")
    #model = nn.DistributedDataParallel(model)
    model = nn.DataParallel(model)
    model.to(device)
    return model

def evaluation(model, loss_func, dataset, device):

    model.eval()
    pred_score = []
    for batch_data in dataset.sample_batch_dev():
        batch_context, batch_mask = batch_data
        context_tensor = torch.tensor(batch_context, dtype=torch.long)
        mask_tensor = torch.tensor(batch_mask, dtype=torch.long)
        output = model(context_tensor, mask_tensor).detach().cpu().numpy()
        pred_score.extend(output)

    dev_input = torch.tensor(np.asarray(pred_score), dtype=torch.float).to(device)
    dev_label = torch.tensor(dataset.dev_label, dtype=torch.float).to(device)

    loss = loss_func(dev_input, dev_label)

    return float(loss.data)


if __name__ == "__main__": 
    config = configparser.RawConfigParser()
    config.read(sys.argv[1])

    gpu_device = config.get("hyperparameters", "gpu_device")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = config.get("data", "ckpt")
    hparam = make_hparam_string(config)
    ckpt_dir = os.path.join(ckpt_dir, hparam)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_path = os.path.join(ckpt_dir, 'train.log')
    best_ckpt_path = os.path.join(ckpt_dir, 'best.ckpt')
    last_ckpt_path = os.path.join(ckpt_dir, 'last.ckpt')

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    epochs = int(config.getfloat("hyperparameters", "max_epochs"))
    learning_rate = config.getfloat("hyperparameters", "learning_rate")
    weight_decay = config.getfloat("hyperparameters", "weight_decay")

    dataset = Dataset(config)

    model = init_model(config, device)
    logger.info(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    #total_parameters = sum(p.numel() for p in parameters)
    #logging.info("| there are totally {} trainable parameters".format(total_parameters))

    for name, param in model.named_parameters():
        print(name)

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(parameters, lr=learning_rate)

    least_loss = 9999999
    
    # model.to(device)
    loss_func.to(device)

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        total_mse = 0
        step = 0
        for batch_data in dataset.sample_batch():

            batch_x, batch_mask,  batch_y = batch_data
         
            context_inputs = torch.tensor(batch_x, dtype=torch.long).to(device)
            mask_inputs = torch.tensor(batch_mask, dtype=torch.long).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)

            output = model(context_inputs, mask_inputs)

            labels = batch_y.to(device)
            mse_loss = loss_func(output, labels) 
            loss = mse_loss #+ weight_decay * norm
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step +=1
            total_loss += float(loss.data) 
            # total_mse += float(mse_loss.data)
            if step % 200 == 0:
                logger.info('| Epoch: {} | step: {} | mse {:5f}'.format(epoch, step, float(mse_loss.data)))

        logger.info('| Epoch: {} | mean mse {:.5f}'.format(epoch, total_mse /step))

        dev_loss = evaluation(model, loss_func, dataset, device)
        if dev_loss < least_loss: 
            least_loss = dev_loss
            # torch.save([model, optimizer, loss_func], ckpt_path)
            save_model = model.module.state_dict().copy()
            torch.save(save_model, best_ckpt_path)
            # torch.save(model.state_dict(), ckpt_path)
            logger.info('| Epoch: {} | mean dev mse: {:.5f} | saved'.format(epoch, dev_loss))
        else:
            save_model = model.module.state_dict().copy()
            #save_model.pop('word_embedding.weight')
            torch.save(save_model, last_ckpt_path)
            logger.info('| Epoch: {} | mean dev mse: {:.5f} |'.format(epoch, dev_loss))
