import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from math import *
import time
from torch.utils import tensorboard
import numpy as np
from model.diffusion_3D.unet import SpatialTransform
import SimpleITK as sitk

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train_oai.json',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    # parse configs
    args = parser.parse_args()

    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    writer = tensorboard.SummaryWriter(opt['path']["tb_logger"])
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    phase = 'train'
    finesize = opt['model']['diffusion']['image_size']
    dataset_opt = opt['datasets']['train']
    batchSize = opt['datasets']['train']['batch_size']
    train_set = Data.create_dataset_OAI(dataset_opt, finesize, phase)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    training_iters = int(ceil(train_set.data_len / float(batchSize)))
    print('Dataset Initialized')

    # model
    diffusion = Model.create_model(opt)
    print("Model Initialized")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']
    if opt['path']['resume_state']:
        print('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    cnter = 0
    while current_epoch < n_epoch:
        current_epoch += 1
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            t = (time.time() - iter_start_time) / batchSize
            # log
            message = '(epoch: %d | iters: %d/%d | time: %.3f) ' % (current_epoch, (istep + 1), training_iters, t)
            errors = diffusion.get_current_log()
            for k, v in errors.items():
                message += '%s: %.6f ' % (k, v)
            print(message)
            if (istep + 1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                t = (time.time() - iter_start_time) / batchSize
                writer.add_scalar("train/l_dif", logs['l_dif'], cnter)
                writer.add_scalar("train/l_sim", logs['l_sim'], cnter)
                writer.add_scalar("train/l_reg", logs['l_reg'], cnter)
                writer.add_scalar("train/l_tot", logs['l_tot'], cnter)
                cnter += 1
        diffusion.scheduler.step()
        writer.add_scalar("lr", diffusion.scheduler.get_last_lr()[0], current_epoch)

        if current_epoch in opt['train']['val_freq']:
            testdataset_opt = {
                "name": "OAI",
                "dataroot": opt['datasets']['test']['dataroot'],
                "data_len": opt['datasets']['test']['data_len']
            }
            test_set = Data.create_dataset_OAI(testdataset_opt, finesize, "test")
            test_loader = Data.create_dataloader(test_set, testdataset_opt, "test")
            stn = SpatialTransform(finesize).cuda()
            
            registSSIM = np.zeros(len(test_set))
            originSSIM = np.zeros(len(test_set))
            
            for istep, test_data in enumerate(test_loader):
                diffusion.feed_data(test_data)
                diffusion.test_registration()
                visuals = diffusion.get_current_registration()
                
                flow = visuals["flow"]
                warp = visuals["warp"]
                
                moving = test_data['M'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
                fixed = test_data['F'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
                regist = stn(moving.type(torch.float32), flow)

                ssim_regist = round(diffusion.netG.loss_ssim(regist, fixed).item(), 4)
                ssim_origin = round(diffusion.netG.loss_ssim(moving, fixed).item(), 4)

                registSSIM[istep] = ssim_regist
                originSSIM[istep] = ssim_origin

                time.sleep(1)

            writer.add_scalar("eval/ssim", np.mean(registSSIM), current_epoch)

        if current_epoch in opt['train']['save_checkpoint_epoch'] or current_epoch == n_epoch:
            diffusion.save_network(current_epoch, current_step)
