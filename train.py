
import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import h5py
import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler

from Network import *  #
from data import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.to(device)
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        smooth = 0.01  
        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target, self.n_classes)
        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input, target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target, dim=(1, 2, 3)) + eps
    union = torch.sum(output, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    output = torch.argmax(output, dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    return dice1, dice2, dice3


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train_loop(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()
    running_loss = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    pbar = tqdm(train_loader)
    for it, (images, masks) in enumerate(pbar):
        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it  
        param_group = optimizer.param_groups[0]
        param_group['lr'] = scheduler[it]
        print(scheduler[it])

        # [b,4,128,128,128] , [b,128,128,128]
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        # outputs = torch.softmax(outputs,dim=1)
        loss = criterion(outputs, masks)
        dice1, dice2, dice3 = cal_dice(outputs, masks)
        pbar.desc = "loss: {:.3f} ".format(loss.item())

        running_loss += loss.item()
        dice1_train += dice1.item()
        dice2_train += dice2.item()
        dice3_train += dice3.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    loss = running_loss / len(train_loader)
    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3}


def val_loop(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs, masks)

            running_loss += loss.item()
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()
            # pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)

    loss = running_loss / len(val_loader)
    dice1 = dice1_val / len(val_loader)
    dice2 = dice2_val / len(val_loader)
    dice3 = dice3_val / len(val_loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3}


def train(model, optimizer, scheduler, criterion, train_loader,
          val_loader, test_loader, epochs, device, train_log, valid_loss_min=999.0):
    best_dice = 0
    best_test_dice = 0
    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(model, optimizer, scheduler, criterion, train_loader, device, e)
        # eval for epoch
        val_metrics = val_loop(model, criterion, val_loader, device)
        # test for epoch
        # test_metrics = val_loop(model, criterion, test_loader, device)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e + 1, epochs, train_metrics["loss"],
                                                                              val_metrics["loss"])
        info2 = "Train--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(train_metrics['dice1'], train_metrics['dice2'],
                                                                  train_metrics['dice3'])
        info3 = "Valid--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(val_metrics['dice1'], val_metrics['dice2'],
                                                                  val_metrics['dice3'])
        # info4 = "Test--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(test_metrics['dice1'],test_metrics['dice2'],test_metrics['dice3'])
        print(info1)
        # print(info2)
        # print(info3)
        with open(train_log, 'a') as f:
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')

        # --------best_epoch and dice----------#
        best_epoch = e + 1
        average_dice = (val_metrics['dice1'] + val_metrics['dice2'] + val_metrics['dice3']) / 3
        if average_dice > best_dice:
            best_dice = average_dice
            with open(args.best_dice_log, 'a') as f:
                f.write(info3)
                # f.write('best_dice: ' + str(best_dice) + ' ' + 'best_epoch: ' + str(best_epoch) + '\n')
                f.write('best_dice: {:.4f}'.format(best_dice) + ' ' + 'best_epoch: {}'.format(best_epoch) + '\n')

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            torch.save(save_file, 'results/model.pth')  #
        else:
            if e + 1 == epochs:
                torch.save(save_file, os.path.join(args.save_path, 'checkpoint{}.pth'.format(e + 1)))
    print("Finished Training!")


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    patch_size = (160, 160, 128)
    train_dataset = BraTS(args.data_path, args.train_txt, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
    val_dataset = BraTS(args.data_path, args.valid_txt, transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))
    test_dataset = BraTS(args.data_path, args.test_txt, transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=12,  # num_worker=4
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                            pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                             pin_memory=True)

    print("using {} device.".format(device))
    # print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))
    print("using {} images for training, {} images for validation, {} images for testing.".format(len(train_dataset),
                                                                                                  len(val_dataset),
                                                                                                  len(test_dataset)))
    model = GBANet(
        in_channels=4,
        n_channels=32,
        n_classes=4,
        # exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        checkpoint_style=None,

    ).to(device)
    model.to(device)

    criterion = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 40])
    scheduler = cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs,
                                 niter_per_ep=len(train_loader), warmup_epochs=args.warmup_epochs,
                                 start_warmup_value=5e-4)

    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        # weight_dict = torch.load(args.weights, map_location=cuda:)
        model.load_state_dict(weight_dict['model'])
        optimizer.load_state_dict(weight_dict['optimizer'])
        print('Successfully loading checkpoint.')

    train(model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader, args.epochs, device,
          train_log=args.train_log)

    if os.path.exists(args.best_weights):
        weight_dict = torch.load(args.best_weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        # optimizer.load_state_dict(weight_dict['optimizer'])
        # print('Successfully loading best_weights.')

    # # metrics1 = val_loop(model, criterion, train_loader, device)
    # metrics2 = val_loop(model, criterion, val_loader, device)
    metrics3 = val_loop(model, criterion, test_loader, device)

    info = "Test--ET: {:.5f} TC: {:.5f} WT: {:.5f} ".format(metrics3['dice1'], metrics3['dice2'], metrics3['dice3'])
    average_test_dice = (metrics3['dice1'] + metrics3['dice2'] + metrics3['dice3']) / 3
    with open(args.best_test_dice_log, 'a') as f:
        f.write(info)
        # f.write('best_dice: ' + str(best_dice) + ' ' + 'best_epoch: ' + str(best_epoch) + '\n')
        f.write('average_test_dice: {:.4f}'.format(average_test_dice) + '\n')


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--min_lr', type=float, default=0.002)
    parser.add_argument('--data_path', type=str, default=r'/home/ww/wlr/datasets/BraTSout')
    parser.add_argument('--train_txt', type=str, default=r'/home/ww/wlr/datasets/BraTSout_txt/train.txt')
    parser.add_argument('--valid_txt', type=str, default=r'/home/ww/wlr/datasets/BraTSout_txt/valid.txt')
    parser.add_argument('--test_txt', type=str, default=r'/home/ww/wlr/datasets/BraTSout_txt/test.txt')
    parser.add_argument('--train_log', type=str, default='results/results.txt')
    parser.add_argument('--best_dice_log', type=str, default='results/best_dice.txt')
    parser.add_argument('--best_test_dice_log', type=str, default='results/best_test_dice.txt')
    parser.add_argument('--weights', type=str, default='checkpoint/checkpoint60.pth')
    parser.add_argument('--best_weights', type=str, default='results/model.pth')
    parser.add_argument('--save_path', type=str, default='checkpoint')

    args = parser.parse_args()

    main(args)
