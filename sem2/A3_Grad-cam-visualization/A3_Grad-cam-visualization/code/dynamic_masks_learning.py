from analysis_settings import img_size_, original_img_path_, mean, std, dynamic_mask_lrs, dynamic_img_batch_,\
     layer_block_, best_loss_, best_epoch_, iteration_epoch_, iteration_epoch_min_, patient_, \
     mask_optimizer_lr_, base_mask_size_list_, tmp_data_path_, single_dynamic_img_batch_, train_max_mask_, mask_count_
import math
import torch.utils.model_zoo as model_zoo
import torchvision
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
from utils.helpers import makedir
import dynamic_masks_module
import numpy as np
import cv2
import matplotlib.pyplot as plt
from find_high_activation import find_high_activation_mask
from utils.preprocess import preprocess_input_function

torchvision.models.resnet50(pretrained=True)

num_classes = 200
episilon = 1e-12
original_img_path = original_img_path_
img_size = img_size_
dynamic_img_batch = dynamic_img_batch_
single_dynamic_img_batch = single_dynamic_img_batch_
img_mask_root_path = './img_mask_root_path'
normalize = transforms.Normalize(mean=mean, std=std)
total_num = 0

def Dynamic_Mask_bb_layer(model):
    original_img_dataset = datasets.ImageFolder(
        original_img_path,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    original_img_loader = torch.utils.data.DataLoader(
        original_img_dataset, batch_size=dynamic_img_batch, shuffle=False,
        num_workers=4, pin_memory=False)

    num_i = 1

    for batch_idx, (x, y) in enumerate(original_img_loader):
        x = x.cuda()
        output = model(x)
        predicted, predicted_index = torch.max(output.data, 1)

        for i in range(x.shape[0]):
            count = 1
            x1_ = x[i]
            y1_ = y[i]
            mask_count = mask_count_
            predicted_current = predicted[i]
            predicted_index_current = predicted_index[i]

            for j in range(mask_count):
                Cal_Mask(x1_, y1_, predicted_current, predicted_index_current, i, count)
                count += 1
                path = os.path.join(tmp_data_path_, 't', 'tmp.jpg')
                original_img = cv2.imread(path)[..., ::-1]
                original_img = original_img / 255
                original_img = original_img.astype('float32')
                x_torch = torch.tensor(original_img)
                x_torch = x_torch.permute(2, 0, 1)
                x_torch = x_torch.unsqueeze(0)
                x2 = preprocess_input_function(x_torch)
                x2 = x2.cuda()
                x1_ = x2
                output1 = model(x1_)
                x1_ = x1_.squeeze(0)

                predicted1, predicted_index1 = torch.max(output1.data, 1)
                predicted_current = predicted1
                predicted_index_current = predicted_index1

            num_i += 1

def Cal_Mask(x1, y1, predicted_i, predicted_index_i, img_i, count):
    img_class = os.listdir(original_img_path)
    print_predict_num = 0
    base_mask_size_list = base_mask_size_list_
    layer_block = layer_block_
    train_max_mask = train_max_mask_

    for base_mask_num in range(len(base_mask_size_list)):
        for base_layer_block in layer_block:
            layer_mask_base = []
            layer_mask = []
            pre_layer_masks = []

            layer_mask.append(np.ones([224, 224]))
            base_mask_size1 = base_mask_size_list[base_mask_num] / base_layer_block
            while (base_layer_block * base_mask_size1 < 224):
                base_mask_size1 = int(base_layer_block * base_mask_size1)
                if(base_mask_size1 > train_max_mask):
                    break
                if (len(pre_layer_masks) == 0):
                    pre_layer_masks.append(np.ones([base_mask_size1, base_mask_size1]))
                mask_img_path = os.path.join(img_mask_root_path, img_class[y1.item()])
                datasets_img_path = os.path.join(original_img_path, img_class[y1.item()])
                img_name = os.listdir(datasets_img_path)
                mask_img_path = os.path.join(mask_img_path, img_name[img_i])

                makedir(mask_img_path)
                c_layer_mask = pre_layer_masks[-1]
                d_mask = dynamic_masks_module.Dynamic_MaskPair(img_size=img_size,
                                                               base_mask_size=base_mask_size1,
                                                               base_layer_block=base_layer_block,
                                                               layer_masks=c_layer_mask)
                d_mask = d_mask.cuda()
                original_act_output = model(x1.unsqueeze(0))
                original_act_predicted, original_act_predicted_index = torch.max(original_act_output.data, 1)

                if(print_predict_num == 0):
                    print(str(count) + "-th:")
                    print("real_label = ", y1)
                    print("original_act_predicted = ", original_act_predicted)
                    print("original_act_predicted_index = ", original_act_predicted_index)
                    print_predict_num += 1

                original_max = predicted_i
                best_loss = best_loss_
                best_epoch = best_epoch_
                iteration_epoch = iteration_epoch_
                iteration_epoch_min = iteration_epoch_min_
                patient = patient_
                mask_optimizer_lr = mask_optimizer_lr_

                base_mask_size = d_mask.base_mask_size
                d_mask = torch.nn.DataParallel(d_mask)
                mask_optimizer_specs = [{'params': d_mask.module.mask, 'lr': mask_optimizer_lr}]
                optimizer = torch.optim.Adam(mask_optimizer_specs)

                for epoch in range(iteration_epoch):
                    if ((epoch - best_epoch) > patient and epoch > iteration_epoch_min):
                        break
                    else:
                        x_mask = d_mask(x1)
                        mask_act_output = model(x_mask)
                        mask_max = mask_act_output[0][predicted_index_i]
                        original_act_patch = original_max
                        mask_act_patch = mask_max
                        act_mse_loss = (mask_act_patch - original_act_patch) ** 2
                        mse_loss = torch.sum(act_mse_loss)
                        mse_loss = mse_loss.cuda()
                        mean_l1_loss = d_mask.module.mask.norm(p=1, dim=(1, 2)) / ((base_mask_size) ** 2)

                        loss = dynamic_mask_lrs['mse'] * mse_loss + dynamic_mask_lrs['l1'] * mean_l1_loss
                        loss = loss.sum()
                        loss = loss.cuda()

                        if (np.min(loss.detach().cpu().numpy()) < best_loss):
                            best_epoch = epoch
                            best_loss = np.min(loss.detach().cpu().numpy())

                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                layer_mask_base.append(d_mask)
                upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
                base_mask = d_mask.module.final_mask.squeeze(0).detach().cpu().numpy()
                pre_layer_masks.append(base_mask)
                real_mask = upsample(d_mask.module.mask.unsqueeze(0)).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                mask_1 = real_mask
                mask_1[:, :, 0] = mask_1[:, :, 0] * layer_mask[-1]

                real_mask_max = np.max(real_mask)
                real_mask_min = np.min(real_mask)
                mask_1 = (real_mask - real_mask_min) / (real_mask_max - real_mask_min)
                layer_mask.append(mask_1[:, :, 0])

                img = x1.permute(1, 2, 0).detach().cpu().numpy()
                mix_img = mask_1 * img
                mix_img = (mix_img - np.min(mix_img)) / (np.max(mix_img) - np.min(mix_img))
                heatmap = cv2.applyColorMap(np.uint8(255 * mask_1), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]
                overlayed_original_img_j = 0.5 * img + 0.3 * heatmap

                overlayed_original_img_j = (overlayed_original_img_j - np.min(overlayed_original_img_j)) / (np.max(overlayed_original_img_j) - np.min(overlayed_original_img_j))
                makedir(os.path.join(mask_img_path, str(base_mask_size_list[base_mask_num]), str(base_layer_block)))
                plt.imsave(os.path.join(mask_img_path, str(base_mask_size_list[base_mask_num]), str(base_layer_block), str(base_mask_size1) + 'base_mask' + 'x' + '.jpg'), base_mask)
                plt.imsave(os.path.join(mask_img_path, str(base_mask_size_list[base_mask_num]), str(base_layer_block), str(base_mask_size1) + 'x' + '.jpg'), mix_img)
                plt.imsave(os.path.join(mask_img_path, str(base_mask_size_list[base_mask_num]), str(base_layer_block), str(base_mask_size1) + 'x' + 'mask' + '.jpg'), real_mask[:, :, 0])
                plt.imsave(os.path.join(mask_img_path, str(base_mask_size_list[base_mask_num]), str(base_layer_block), str(base_mask_size1) + 'x' + 'mix' + '.jpg'), overlayed_original_img_j)

def Generate_Mask(num_i, count):
    mdm_save = './dm_result'
    original_path = './original_datasets'
    mask_path = img_mask_root_path
    percentile_list = [1]
    base_mask_size_list = base_mask_size_list_
    layer_block = layer_block_
    img_size = 224
    result_path = './Result'
    tmp_data_path = tmp_data_path_
    class_path = os.listdir(mask_path)
    makedir(result_path)
    all_num = 0

    for percentile in percentile_list:
        num = 0
        num_i_n = 0

        for every_class_path in class_path:
            img_path = os.listdir(os.path.join(mask_path, every_class_path))

            for every_img_path in img_path:
                single_img_path = os.path.join(mask_path, every_class_path, every_img_path)
                num_i_n += 1

                if(num_i_n == num_i):
                    all_mask = np.zeros([img_size, img_size])
                    real_all_mask = np.zeros([224, 224])

                    for base_size in base_mask_size_list:
                        for j in range(len(layer_block)):
                            ig_name = os.listdir(os.path.join(single_img_path, str(base_size), str(layer_block[j])))
                            for k in range(int(len(ig_name) / 4)):
                                mask_img = cv2.imread(os.path.join(single_img_path, str(base_size), str(layer_block[j]), str(int(base_size * math.pow(layer_block[j], k))) + 'xmask.jpg'), cv2.IMREAD_GRAYSCALE)
                                mask_img = mask_img - np.amin(mask_img)
                                mask_img = mask_img / np.amax(mask_img)
                                mask_img_binary = find_high_activation_mask(mask_img, percentile=percentile)
                                real_all_mask += mask_img
                                all_mask += mask_img_binary
                                all_num += 1

                    heat_map = all_mask
                    heat_map = heat_map - np.amin(heat_map)
                    heat_map = heat_map / np.amax(heat_map)
                    heat_map = np.reshape(heat_map, (224, 224, 1))
                    heat_map_binary = find_high_activation_mask(heat_map, percentile=75)
                    heat_map_binary[heat_map_binary > 0] = 1

                    original_img = cv2.imread(os.path.join(original_path, every_class_path, every_img_path))[..., ::-1]

                    real_heat_map = real_all_mask
                    real_heat_map = real_heat_map - np.amin(real_heat_map)
                    real_heat_map = real_heat_map / np.amax(real_heat_map)

                    heatmaps = cv2.applyColorMap(np.uint8(255 * real_heat_map), cv2.COLORMAP_JET)
                    heatmaps = np.float32(heatmaps) / 255
                    heatmaps = heatmaps[..., ::-1]
                    overlayed_original_img_j = 0.5 * original_img / 255 + 0.3 * heatmaps
                    overlayed_original_img_j = (overlayed_original_img_j - np.min(overlayed_original_img_j)) / (
                                np.max(overlayed_original_img_j) - np.min(overlayed_original_img_j))

                    real_heat_map = np.reshape(real_heat_map, (224, 224, 1))
                    real_heat_map = real_heat_map.repeat(3, 2)
                    real_heat_map = 1 - real_heat_map
                    num += 1

                    makedir(os.path.join(mdm_save, every_class_path, every_img_path))
                    plt.imsave(os.path.join(mdm_save, every_class_path, every_img_path, 'camcam' + str(count) + '.jpg'), overlayed_original_img_j)
                    plt.imsave(os.path.join(mdm_save, every_class_path, every_img_path, 'mask' + str(count) + '.jpg'), real_heat_map[:, :, 0])

                    img_list = os.listdir(os.path.join(mdm_save, every_class_path, every_img_path))
                    mask_current = np.zeros([224, 224])

                    for b in range(int(len(img_list)/4 + 1)):
                        tmp_single_mask = 1 - cv2.imread(os.path.join(mdm_save, every_class_path, every_img_path, 'mask' + str(b+1) + '.jpg'), cv2.IMREAD_GRAYSCALE) / 255
                        mask_current += tmp_single_mask

                    mask_current += 1 - real_heat_map[:, :, 0]
                    mask_current = mask_current - np.min(mask_current)
                    mask_current = mask_current / np.max(mask_current)
                    mask_current = 1 - mask_current

                    mask_current_rgb = np.zeros([224, 224, 3])
                    mask_current_rgb[:, :, 0] = mask_current
                    mask_current_rgb[:, :, 1] = mask_current
                    mask_current_rgb[:, :, 2] = mask_current

                    real_original_heat_map_ = mask_current_rgb * original_img
                    real_original_heat_map_ = real_original_heat_map_.astype('uint8')

                    plt.imsave(os.path.join(mdm_save, every_class_path, every_img_path, 'camcam' + '_r' + str(count) + '.jpg'), real_original_heat_map_)
                    plt.imsave(os.path.join(mdm_save, every_class_path, every_img_path, 'mask' + '_r' + str(count) + '.jpg'), real_heat_map)

                    makedir(os.path.join(tmp_data_path, 't'))
                    plt.imsave(os.path.join(tmp_data_path, 't', 'tmp' + '.jpg'), real_original_heat_map_)

    return tmp_data_path

if __name__ == '__main__':
    load_model_path = './pretrained_models/resnet50/' # Trained model file
    resnet = torch.load(load_model_path).eval().cuda()
    model = resnet.cuda()
    model_multi = torch.nn.DataParallel(model)

    Dynamic_Mask_bb_layer(model_multi)

