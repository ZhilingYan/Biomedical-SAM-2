import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import cfg
from func.utils import *
import pandas as pd
import monai


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mask_type = torch.float32
iou_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')  
dicece_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

torch.backends.cudnn.benchmark = True


def train_sam2(args, net: nn.Module, optimizer, train_loader, epoch, writer):
    
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    
    # train mode
    net.train()
    optimizer.zero_grad()

    # init
    epoch_loss = 0
    memory_bank_list = []
    lossfunc = criterion_G
    feat_sizes = [(256, 256), (128, 128), (64, 64)]


    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            
            to_cat_memory = []
            to_cat_memory_pos = []

            # input image and gt masks
            imgs = pack['image'].to(dtype = mask_type, device = GPUdevice)
            masks = pack['mask'].to(dtype = mask_type, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            # click prompt: unsqueeze to indicate only one click, add more click across this dimension
            if 'pt' in pack:
                pt_temp = pack['pt'].to(device = GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device = GPUdevice) # tensor([1, 1, 1, 1], device='cuda:0')
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None

            '''Train image encoder'''                    
            backbone_out = net.forward_image(imgs)
            # backbone_out["backbone_fpn"]:
            #torch.Size([batch, 32, 256, 256])
            #torch.Size([batch, 64, 128, 128])
            #torch.Size([batch, 256, 64, 64])
            #backbone_out["vision_pos_enc"]:
            #torch.Size([batch, 256, 256, 256])
            #torch.Size([batch, 256, 128, 128])
            #torch.Size([batch, 256, 64, 64])
            #*********************
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            # dimension hint for your future use
            # vision_feats: list: length = 3
            # vision_feats[0]: torch.Size([65536, batch, 32])
            # vision_feats[1]: torch.Size([16384, batch, 64])
            # vision_feats[2]: torch.Size([4096, batch, 256])
            # vision_pos_embeds[0]: torch.Size([65536, batch, 256])
            # vision_pos_embeds[1]: torch.Size([16384, batch, 256])
            # vision_pos_embeds[2]: torch.Size([4096, batch, 256])
            
            

            '''Train memory attention to condition on meomory bank'''         
            B = vision_feats[-1].size(1)  # batch size 
            
            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                
            else:
                for element in memory_bank_list:
                    to_cat_memory.append((element[0]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_features
                    to_cat_memory_pos.append((element[1]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_pos_enc
                memory = torch.cat(to_cat_memory, dim=0)
                memory_pos = torch.cat(to_cat_memory_pos, dim=0)

                
                memory = memory.repeat(1, B, 1) 
                memory_pos = memory_pos.repeat(1, B, 1) 


                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                    )


            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            
            # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
            # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
            # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed


            '''prompt encoder'''         
            with torch.no_grad():
                B,_, H, W = masks.shape
                if args.nprompt == 'click':
                    points=(coords_torch, labels_torch)
                    flag = True

                    se, de = net.sam_prompt_encoder(
                        points=points, #(coords_torch, labels_torch)
                        boxes=None,
                        masks=None,
                        batch_size=B,
                    )
            # dimension hint for your future use
            # se: torch.Size([batch, n+1, 256])
            # de: torch.Size([batch, 256, 64, 64])
                elif args.nprompt == 'bbox':
                    boxes_torch = torch.zeros((B, 4), dtype=torch.float32, device=device)

                    for i in range(B):
                        mask = masks[i, 0]  # shape: (H, W)
                        non_zero_indices = torch.nonzero(mask)

                        if non_zero_indices.numel() > 0:
                            y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
                            y_max, x_max = torch.max(non_zero_indices, dim=0)[0]

                            w = x_max.item() - x_min.item()
                            h = y_max.item() - y_min.item()

                            mid_x = (x_min.item() + x_max.item()) / 2
                            mid_y = (y_min.item() + y_max.item()) / 2

                            num_rand = np.random.randn(2) * 0.1 
                            w *= 1 + num_rand[0]
                            h *= 1 + num_rand[1]

                            x_min_new = mid_x - w / 2
                            x_max_new = mid_x + w / 2
                            y_min_new = mid_y - h / 2
                            y_max_new = mid_y + h / 2

                            x_min_new = max(0, x_min_new)
                            x_max_new = min(W, x_max_new)
                            y_min_new = max(0, y_min_new)
                            y_max_new = min(H, y_max_new)

                            boxes_torch[i] = torch.tensor([x_min_new, y_min_new, x_max_new, y_max_new], dtype=torch.float32, device=device)                       
                        else:
                            boxes_torch[i] = torch.tensor([0, 0, W, H], dtype=torch.float32, device=device)
                
                    
                    se, de = net.sam_prompt_encoder(
                        points=None,
                        boxes=boxes_torch,
                        masks=None,
                        batch_size=B,
                    )
                    flag = False


            
            '''train mask decoder'''       
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    high_res_features = high_res_feats,
                )
            # dimension hint for your future use
            # low_res_multimasks: torch.Size([batch, multimask_output, 256, 256])
            # iou_predictions.shape:torch.Size([batch, multimask_output])
            # sam_output_tokens.shape:torch.Size([batch, multimask_output, 256])
            # object_score_logits.shape:torch.Size([batch, 1])
            
            
            # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            

            '''memory encoder'''       
            # new caluculated memory features
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=flag)  
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
                
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)


            # add single maskmem_features, maskmem_pos_enc, iou
            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                             (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                             iou_predictions[batch, 0]])
            
            else:
                for batch in range(maskmem_features.size(0)):
                    
                    # current simlarity matrix in existing memory bank
                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                    # normalise
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                         memory_bank_maskmem_features_norm.t())

                    # replace diagonal (diagnoal always simiarity = 1)
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    # first find the minimum similarity from memory feature and the maximum similarity from memory bank
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores) 
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    # replace with less similar object
                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        # soft iou, not stricly greater than current iou
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index) 
                            memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions[batch, 0]])

            # backpropagation
            if not args.DiceCEloss:
                loss = lossfunc(pred, masks)
            else:
                diceloss = dicece_loss(pred, masks)                
                #iouloss = iou_loss(iou_predictions, labels_torch)
                loss = diceloss            
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

            pbar.update()

    return epoch_loss/len(train_loader)




def validation_sam2(args, val_loader, epoch, net: nn.Module, clean_dir=True):

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # eval mode
    net.eval()

    n_val = len(val_loader) 
    threshold = [0.5]
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['mask'].to(dtype = torch.float32, device = GPUdevice)

            
            if 'pt' in pack:
                pt_temp = pack['pt'].to(device = GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device = GPUdevice) # tensor([1, 1, 1, 1], device='cuda:0')
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None



            '''test'''
            with torch.no_grad():

                """ image encoder """
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1) 

                """ memory condition """
                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")

                else:
                    for element in memory_bank_list:
                        maskmem_features = element[0]
                        maskmem_pos_enc = element[1]
                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                    memory = torch.cat(to_cat_memory, dim=0)
                    memory_pos = torch.cat(to_cat_memory_pos, dim=0)

                    memory = memory.repeat(1, B, 1) 
                    memory_pos = memory_pos.repeat(1, B, 1) 

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                        )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                """ prompt encoder """
                B,_, H, W = masks.shape
                if args.nprompt == 'click':
                    points = (coords_torch, labels_torch)
                    flag = True

                    se, de = net.sam_prompt_encoder(
                        points=points, 
                        boxes=None,
                        masks=None,
                        batch_size=B,
                    )
                else:
                    boxes_torch = torch.zeros((B, 4), dtype=torch.float32, device=device)

                    for i in range(B):
                        mask = masks[i, 0]  # shape: (H, W)
                        non_zero_indices = torch.nonzero(mask)

                        if non_zero_indices.numel() > 0:
                            y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
                            y_max, x_max = torch.max(non_zero_indices, dim=0)[0]

                            w = x_max.item() - x_min.item()
                            h = y_max.item() - y_min.item()

                            mid_x = (x_min.item() + x_max.item()) / 2
                            mid_y = (y_min.item() + y_max.item()) / 2

                            num_rand = np.random.randn(2) * 0.1 
                            w *= 1 + num_rand[0]
                            h *= 1 + num_rand[1]

                            x_min_new = mid_x - w / 2
                            x_max_new = mid_x + w / 2
                            y_min_new = mid_y - h / 2
                            y_max_new = mid_y + h / 2

                            x_min_new = max(0, x_min_new)
                            x_max_new = min(W, x_max_new)
                            y_min_new = max(0, y_min_new)
                            y_max_new = min(H, y_max_new)

                            boxes_torch[i] = torch.tensor([x_min_new, y_min_new, x_max_new, y_max_new], dtype=torch.float32, device=device)
                        else:
                            boxes_torch[i] = torch.tensor([0, 0, W, H], dtype=torch.float32, device=device)
                                
                    se, de = net.sam_prompt_encoder(
                        points=None,
                        boxes=boxes_torch,
                        masks=None,
                        batch_size=B,
                    )
                    flag = False

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, 
                    repeat_image=False,  
                    high_res_features = high_res_feats,
                )

                # prediction
                pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            
                """ memory encoder """
                maskmem_features, maskmem_pos_enc = net._encode_new_memory( 
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks,
                    is_mask_from_pts=flag)  
                    
                maskmem_features = maskmem_features.to(torch.bfloat16)
                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)


                """ memory bank """
                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
                                                 iou_predictions[batch, 0]])
                
                else:
                    for batch in range(maskmem_features.size(0)):
                        
                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                             memory_bank_maskmem_features_norm.t())

                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores) 
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                        if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index) 
                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
                                                         iou_predictions[batch, 0]])


                # binary mask and calculate loss, iou, dice
                if not args.DiceCEloss:
                    loss = lossfunc(pred, masks)
                else:
                    diceceloss = dicece_loss(pred, masks)                
                    #iouloss = iou_loss(iou_predictions, labels_torch)
                    loss = diceceloss
                total_loss += loss
                pred = (pred> 0.5).float()
                eiou, edice = eval_seg(pred, masks, threshold)
                total_eiou += eiou
                total_dice += edice

                '''vis images'''
                if ind % args.vis == 0:
                    namecat = 'Test'
                    for na in name:
                        img_name = na
                        namecat = namecat + img_name + '+'
                    vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=None)
                            
            pbar.update()

    return total_loss/ n_val , tuple([total_eiou/n_val, total_dice/n_val])

