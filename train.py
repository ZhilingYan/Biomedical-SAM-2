# train.py
#!/usr/bin/env	python3

""" 
train network using pytorch
adapted from https://github.com/MedicineToken/Medical-SAM2/tree/main
"""

import os
import time
from datetime import datetime

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import cfg
import func.function as function
from func.dataset import *
from func.utils import *

TIME_NOW = datetime.now().strftime("%F_%H-%M-%S.%f")
LOG_DIR = 'runs'

def main():
    # use bfloat16 for the entire work
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    # optimisation
    for param in net.image_encoder.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                           lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
    #                        lr=args.lr, weight_decay=0, momentum=0.99)                           
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) 

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)


    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    
    # example of BUSI dataset
    if args.dataset == 'BUSI':
        '''BUSI data'''
        refuge_train_dataset = BUSI(args, args.data_path, transform = transform_train, mode = 'Training', train_file_dir="BUSI_train.txt", val_file_dir="BUSI_val.txt")
        refuge_test_dataset = BUSI(args, args.data_path, transform = transform_test, mode = 'Test', train_file_dir="BUSI_train.txt", val_file_dir="BUSI_val.txt")

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''


    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join('checkpoint', args.net, TIME_NOW)
    #use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            LOG_DIR, args.net, TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0

    if args.model_name == 'biomedical-sam2':
        for epoch in range(args.epoch):

            # training
            net.train()
            time_start = time.time()
            loss = function.train_sam2(args, net, optimizer, nice_train_loader, epoch, writer)
            logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
            time_end = time.time()
            print('time_for_training ', time_end - time_start)

            # validation
            net.eval()
            if epoch % args.val_freq == 0 or epoch == args.epoch-1:

                tol, (eiou, edice) = function.validation_sam2(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

                if args.distributed != 'none':
                    sd = net.module.state_dict()
                else:
                    sd = net.state_dict()

                if  edice > best_dice:
                    best_dice = edice
                    best_tol = tol
                    is_best = True

                    save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'best_tol': best_tol,
                    'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
                else:
                    is_best = False

        writer.close()  
       

if __name__ == '__main__':
    main()
