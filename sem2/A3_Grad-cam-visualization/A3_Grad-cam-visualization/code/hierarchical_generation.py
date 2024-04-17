from utils.preprocess import preprocess_input_function
from analysis_settings import img_size_, original_img_path_, mean, std, dynamic_img_batch_, mix_iteration_check_, \
    single_dynamic_img_batch_, mix_mask_optimizer_lr_, mix_iteration_epoch_, mask_count_
import torch.utils.model_zoo as model_zoo
import torchvision
import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
from utils.helpers import makedir
import numpy as np
import cv2
import matplotlib.pyplot as plt
from find_high_activation import find_high_activation_mask_y

torchvision.models.resnet50(pretrained=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(os.environ['CUDA_VISIBLE_DEVICES'])

num_prototypes = 2000
num_classes = 200
num_prototypes_per_class = int(num_prototypes // num_classes)

episilon = 1e-12
original_img_path = original_img_path_
img_size = img_size_
dynamic_img_batch = dynamic_img_batch_
single_dynamic_img_batch = single_dynamic_img_batch_
img_mask_root_path = './img_mask_root_path'
normalize = transforms.Normalize(mean=mean, std=std)

original_image_path = './original_datasets'
mask_image_path = './dm_result'
class_list = os.listdir(mask_image_path)

load_model_path = './pretrained_models/resnet50/' # Trained model file
resnet = torch.load(load_model_path).eval().cuda()
model = resnet.cuda()
model_multi = torch.nn.DataParallel(model)
class_label = 0

for i in class_list:
    class_name = class_list = os.listdir(os.path.join(mask_image_path, i))

    for j in class_name:

        mask_list = []
        mask_weight = []
        mask_img_list = []
        r = os.listdir(os.path.join(mask_image_path, i, j))

        original_image = cv2.imread(os.path.join(original_image_path, i, j))[..., ::-1]

        for k in range(int(len(r)/4)):
            mask_k = cv2.imread(os.path.join(mask_image_path, i, j, 'mask' + str(k + 1) + '.jpg'), cv2.IMREAD_GRAYSCALE) / 255
            mask_k = mask_k - np.min(mask_k)
            mask_k = mask_k / np.max(mask_k)
            mask_k = 1 - mask_k
            mask_k_binary, threshold = find_high_activation_mask_y(mask_k, 85)
            mask_k = mask_k_binary * mask_k
            mask_k = mask_k - threshold
            mask_k[mask_k < 0] = 0
            mask_k = mask_k / np.max(mask_k)

            plt.imshow(mask_k)
            plt.show()

            mask_list.append(mask_k)

            mask_k_rgb = np.zeros([224, 224, 3])
            mask_k_rgb[:, :, 0] = mask_k
            mask_k_rgb[:, :, 1] = mask_k
            mask_k_rgb[:, :, 2] = mask_k

            mask_img = original_image * mask_k_rgb
            mask_img = mask_img.astype('uint8')

            makedir(os.path.join('tmp_mix_img', 'current'))
            plt.imsave(os.path.join('tmp_mix_img', 'current', 'tmp.jpg'), mask_img)

            original_img_dataset1 = datasets.ImageFolder(
                os.path.join('tmp_mix_img'),
                transforms.Compose([
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.ToTensor(),
                    normalize,
                ]))

            original_img_loader1 = torch.utils.data.DataLoader(
                original_img_dataset1, batch_size=single_dynamic_img_batch, shuffle=False,
                num_workers=4, pin_memory=False)

            for batch_idx, (x2, y2) in enumerate(original_img_loader1):
                x2 = x2.cuda()[0]
                logits = model_multi(x2.unsqueeze(0))
                logits = torch.softmax(logits, dim=-1)
                activation = logits[0][class_label].item()
                mask_weight.append(activation)

        original_img = cv2.imread(os.path.join(original_image_path, i, j))[..., ::-1]
        original_img = original_img / 255
        original_img = original_img.astype('float32')
        original_image_torch = torch.tensor(original_img)
        original_image_torch = original_image_torch.permute(2, 0, 1)
        original_image_torch = original_image_torch.unsqueeze(0)
        x_torch = preprocess_input_function(original_image_torch)
        x_torch = x_torch.cuda()

        logits_real = model_multi(x_torch)
        real_predicted, real_predicted_index = torch.max(logits_real.data, 1)
        real_predicted_index = real_predicted_index.item()
        logits_2 = real_predicted

        mix_iteration_epoch = mix_iteration_epoch_
        learning_masks = nn.Parameter(torch.tensor(np.array(mask_list).astype('float32')).cuda(), requires_grad=False)
        learning_final_mask = np.zeros([224, 224])
        mix_mask_optimizer_lr = mix_mask_optimizer_lr_
        mix_iteration_check = mix_iteration_check_
        learning_logits = nn.Parameter(torch.tensor(np.ones([len(mask_list)]).astype('float32')).cuda(), requires_grad=True)

        mask_count = mask_count_
        v_parameter = np.zeros([mask_count, mask_count])
        for v_i in range(mask_count):
            for v_j in range(v_i + 1):
                v_parameter[v_i][v_j] = 1

        Increase_parameter = nn.Parameter(torch.tensor(np.array(v_parameter).astype('float32')).cuda(), requires_grad=False)
        mix_mask_optimizer_specs = [{'params': learning_logits, 'lr': mix_mask_optimizer_lr}]
        optimizer = torch.optim.Adam(mix_mask_optimizer_specs)

        original_img_t = cv2.imread(os.path.join(original_image_path, i, j))[..., ::-1]
        original_img_t = original_img_t / 255
        original_img_t = original_img_t.astype('float32')
        original_img_t2 = torch.tensor(original_img_t).cuda()

        for epoch in range(mix_iteration_epoch):
            learning_logits1 = ((learning_logits * learning_logits) @ Increase_parameter).softmax(dim=-1)
            final_mask = learning_masks * learning_logits1.unsqueeze(-1).unsqueeze(-1)
            base_final_mask = torch.sum(final_mask, dim=0)
            base_final_mask = base_final_mask - torch.min(base_final_mask)
            base_final_mask = base_final_mask / torch.max(base_final_mask)

            mask_original_image_torch1 = base_final_mask.unsqueeze(-1) * original_img_t2
            mask_original_image_torch_activation2 = mask_original_image_torch1.permute(2, 0, 1)
            mask_original_image_torch_activation3 = mask_original_image_torch_activation2.unsqueeze(0)
            x_mask_original_image_torch_activation4 = preprocess_input_function(mask_original_image_torch_activation3)
            x_mask_original_image_torch_activation5 = x_mask_original_image_torch_activation4.cuda()

            logits_1 = model_multi(x_mask_original_image_torch_activation5)[0][real_predicted_index]

            mask_logits = (logits_1 - logits_2) ** 2
            mask_activation = mask_logits

            if (epoch % mix_iteration_check == 0):
                print("learning_logits1 = ", learning_logits1)
                print("mask_logits = ", mask_logits)

            loss = mask_activation
            loss = loss.sum()
            learning_final_mask = base_final_mask.detach().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        heatmaps = cv2.applyColorMap(np.uint8(255 * learning_final_mask), cv2.COLORMAP_JET)
        heatmaps = np.float32(heatmaps) / 255
        heatmaps = heatmaps[..., ::-1]
        overlayed_original_img_j = 0.5 * original_image / 255 + 0.3 * heatmaps

        makedir(os.path.join('hdm_saliency_maps', i, j))
        plt.imsave(os.path.join('hdm_saliency_maps', i, j, 'mask.jpg'), learning_final_mask)
        plt.imsave(os.path.join('hdm_saliency_maps', i, j, 'camcam.jpg'), overlayed_original_img_j)

    class_label += 1