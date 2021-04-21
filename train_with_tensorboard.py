# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils1.config import Config
from nets1.yolo_training import YOLOLoss, Generator
from nets1.yolo3 import YoloBody
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm


def fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0
    for iteration in range(epoch_size):
        start_time = time.time()
        images, targets = next(gen)
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        # 将loss写入tensorboard，每一步都写
        writer.add_scalar('Train_loss', loss, (epoch * epoch_size + iteration))

        total_loss += loss
        waste_time = time.time() - start_time
        print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (
        total_loss / (iteration + 1), waste_time))

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration in range(epoch_size_val):
            images_val, targets_val = next(gen_val)

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)


    # 将loss写入tensorboard，每个世代保存一次
    writer.add_scalar('Val_loss',val_loss/(epoch_size_val+1), epoch)
    print('Finish Validation')
    print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


#    torch.save(model.state_dict(), 'logs/hao-Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)),_use_new_zipfile_serialization=False)

if __name__ == "__main__":
    # 参数初始化
    input_shape = (416, 416)
    annotation_path = '2007_train.txt'
    model = YoloBody(Config)
    Cuda = True

    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model_data/yolo_weights.pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

   #tensorboard
    writer = SummaryWriter(log_dir='logs',flush_secs=60)


    if Cuda:
        graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor).cuda()
    else:
        graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor)
    writer.add_graph(model, (graph_inputs,))

    if True:
        # 最开始使用1e-3的学习率可以收敛的更快
        lr = 1e-3
        Batch_size = 6
        Init_Epoch = 0
        Freeze_Epoch = 25

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        gen = Generator(Batch_size, lines[:num_train],
                        (Config["img_h"], Config["img_w"])).generate()
        gen_val = Generator(Batch_size, lines[num_train:],
                            (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size






        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 6
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        gen = Generator(Batch_size, lines[:num_train],
                        (Config["img_h"], Config["img_w"])).generate()
        gen_val = Generator(Batch_size, lines[num_train:],
                            (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()




            # yolo2yolo3
            # voc_annonations