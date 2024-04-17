train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

base_architecture = 'resnet50'
img_size = 224
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = '003'

data_path = './original_datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

original_img_path_ = './original_datasets'
tmp_data_path_ = './tmp_original_datasets'
mix_tmp_data_path_ = './mix_tmp_original_datasets'
img_size_ = 224
dynamic_img_batch_ = 2

dynamic_mask_lrs = {'mse': 1e-2,
                    'l1': 1}

base_mask_size_ = 10
best_loss_ = 1e8
best_epoch_ = -1
best_mask_position_ = -1
iteration_epoch_ = 2000
iteration_epoch_min_ = 800
patient_ = 500
mask_optimizer_lr_ = 1e-2
check_epoch_ = 50
mix_mask_optimizer_lr_ = 1e-1
single_dynamic_img_batch_ = 1

dynamic_mask_batch_ = 80
max_mask_ = 100
mask_count_ = 3
train_max_mask_ = 50
mix_iteration_epoch_ = 400
mix_iteration_check_ = 40
d_mix_mask_optimizer_lr_ = 3e-2
d_mix_iteration_epoch_ = 400
d_mix_iteration_check_ = 20

add_iteration_epoch_ = 200
add_mask_optimizer_lr_ = 3e-2
add_iteration_check_ = 20

mix_mask_lrs = {'activation': 1e-1,
                'l1': 1}

mix_mask_lrs1_ = {'mse': 1e-1,
                    'l1': 3}

mix_mask_lrs2_ = {'mse': 1e-1,
                    'l1': 3}

final_mix_mask_lrs_ = {'mse': 2e-1,
                    'l1': 1}

base_mask_size_list_ = [6, 7, 8, 9, 10, 11]
layer_block_ = [2, 3, 5]