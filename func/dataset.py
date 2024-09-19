import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import tifffile as tiff

from func.utils import random_click
        
class BUSI(Dataset):
    def __init__(
        self, 
        args, 
        data_path , 
        transform = None, 
        transform_msk = None, 
        mode = 'Training',
        prompt = 'click', 
        plane = False,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
    ):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk
        
        self.sample_list = []

        if self.mode == "Training":
            with open(os.path.join(self.data_path, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.mode == "Test":
            with open(os.path.join(self.data_path, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.mode))

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, index):
    
        case = self.sample_list[index]

        # raw image and raters path
        img_path = os.path.join(self.data_path, 'Dataset_BUSI_with_GT', case + '.png')
        multi_rater_cup_path = os.path.join(self.data_path, 'Dataset_BUSI_with_GT', case + '_mask.png')

        # raw image and rater images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = Image.open(multi_rater_cup_path).convert('L')

        # apply transform
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = torch.as_tensor((self.transform(multi_rater_cup) >=0.5).float(), dtype=torch.float32)
            #multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)


        point_label_cup, pt_cup = random_click(np.array((multi_rater_cup.mean(axis=0)).squeeze(0)), point_label = 1)
        selected_rater_mask_cup_ori = multi_rater_cup.mean(axis=0)
        selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 


        selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0).unsqueeze(1), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
        selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


        image_meta_dict = {'filename_or_obj':case}
        return {
            'image':img,
            'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'mask': selected_rater_mask_cup, 
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }

