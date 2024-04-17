import torch.utils.model_zoo as model_zoo
import torchvision
import os
import csv
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
import utils.save as save
import time
import datetime
from utils.log import create_logger
from timm import create_model as creat

# train resnet50
torchvision.models.resnet50(pretrained=True)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
img_size = 224
train_dir = './original_datasets/train_cropped_augmented/'
test_dir = './original_datasets/test_cropped/'
n_examples = 0
n_correct = 0
train_batch_size = 60
test_batch_size = 20
is_train = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    base_architecture = 'resnet50'
    data_log = '1'
    model_dir = './saved_models/' + base_architecture + '/' + data_log + '/'
    datasets_name_txt = 'CUB_200_2011'
    result_path = './Result/'
    train_test_path = 'Train_and_Test'
    time_path = str(datetime.datetime.now()).replace(':', '.')
    makedir(os.path.join(result_path, time_path))
    makedir(model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    training_set_size_txt = len(train_loader.dataset)
    test_set_size_txt = len(test_loader.dataset)

    print('training set size: {0}'.format(training_set_size_txt))
    print('test set size: {0}'.format(test_set_size_txt))
    print('train batch size: {0}'.format(train_batch_size))
    print('test batch size: {0}'.format(test_batch_size))

    with open(os.path.join(result_path, time_path, datasets_name_txt) + str('.txt'), 'w') as f:
        f.write("datasets_name:" + datasets_name_txt)
        f.write("\n")
        f.write("training_set_size_txt = " + str(training_set_size_txt))
        f.write("\n")
        f.write("test_set_size_txt = " + str(test_set_size_txt))
        f.write("\n")
        f.write("train batch size = " + str(train_batch_size))
        f.write("\n")
        f.write("test batch size = " + str(test_batch_size))
        f.write("\n")

    with open(os.path.join(result_path, time_path, train_test_path) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['epoch', 'train_time', 'avg_train_loss', 'train_acc', 'test_time', 'test_acc'])

    model = creat('resnet50', pretrained=True, num_classes=200)
    model = model.cuda()
    criteon = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        train_correct = 0
        train_num = 0
        total_train_loss = 0
        train_start = time.time()
        for batchidx, (x, label) in enumerate(train_loader):
            x = x.cuda()
            label = label.cuda()
            logits = model(x)
            loss = criteon(logits, label)
            pred = logits.argmax(dim=1)
            train_correct += torch.eq(pred, label).float().sum().item()
            train_num += x.size(0)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_end = time.time()
        avg_train_loss = total_train_loss / train_num
        train_time = train_end - train_start
        train_acc = train_correct / train_num

        print("epoch :", epoch)
        print("train time:", train_time)
        print("avg_train_loss :", avg_train_loss)
        print("train_acc:", train_acc)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            test_start = time.time()
            for x, label in test_loader:
                x, label = x.cuda(), label.cuda()

                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            test_end = time.time()
            test_time = test_end - test_start
            print("test time:", test_time)
            test_acc = total_correct / total_num
            print("test_acc :", test_acc)

            save.save_model_w_condition(model=model, model_dir=model_dir, model_name=str(epoch) + '',
                                        accu=test_acc,
                                        target_accu=0.60, log=log)

        result = []
        result.append(str(epoch))
        result.append(str(train_time))
        result.append(str(avg_train_loss))
        result.append(str(train_acc))
        result.append(str(test_time))
        result.append(str(test_acc))

        with open(os.path.join(result_path, time_path ,train_test_path) + '.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(result)

if __name__ == '__main__':
    main()