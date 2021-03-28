# coding=utf-8
import datetime
import math
import os
import shutil
import gc
import time

import numpy as np
import torch
from torch.autograd import Variable

from utils import utils
import tqdm

class Trainer(object):

    def __init__(self, cmd, cuda, model, criterion, optimizer,
                 train_loader, val_loader, log_file, max_epoch,
                 interval_validate=None, lr_scheduler=None,
                 timestamp='2020-10-15', checkpoint_dir=None, print_freq=1):
        """
        :param cuda:
        :param model:
        :param optimizer:
        :param train_loader:
        :param val_loader:
        :param log_file: log file name. logs are appended to this file.
        :param max_iter:
        :param interval_validate:
        :param checkpoint_dir:
        :param lr_scheduler:
        """

        self.cmd = cmd
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now()

        if cmd == 'train':
            self.interval_validate = len(self.train_loader) if interval_validate is None else interval_validate

        self.epoch = 0
        self.max_epoch = max_epoch

        self.iteration = 0

        # self.max_iter = max_iter
        self.best_top1 = 0
        self.best_top5 = 0
        self.print_freq = print_freq

        self.timestamp= timestamp
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file


    def print_log(self, log_str):
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')


    def train(self):

        # int(math.ceil(1. * self.max_iter / len(self.train_loader))) # 117
        print("train max epoch {0}".format(self.max_epoch))
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Train', ncols=80):
            self.epoch = epoch

            epoch_time = utils.AverageMeter()
            batch_time = utils.AverageMeter()
            data_time = utils.AverageMeter()
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()


            self.model.train()
            self.optim.zero_grad()

            epoch_end = time.time()
            iter_end = time.time()

            for batch_idx, (imgs, target) in enumerate(self.train_loader):

                iteration = batch_idx + self.epoch * len(self.train_loader)
                # print('\n{} {} {}'.format(batch_idx, self.epoch, len(self.train_loader)))
                data_time.update(time.time() - iter_end)

                gc.collect()  # 清理内存

                if self.iteration != 0 and (iteration - 1) != self.iteration:
                    continue  # for resuming
                self.iteration = iteration

                if (self.iteration + 1) % self.interval_validate == 0:
                    if self.val_loader:
                        self.validate()
                if self.cuda:
                    imgs, target = imgs.cuda(), target.cuda(async=True)
                # imgs, target = Variable(imgs), Variable(target)

                output = self.model(imgs)
                loss = self.criterion(output, target)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while training')

                # measure accuracy and record loss
                prec1, prec5 = utils.accuracy(output.data, target.data, topk=(1, 5))
                losses.update(loss.item(), imgs.size(0))
                top1.update(prec1[0], imgs.size(0))
                top5.update(prec5[0], imgs.size(0))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # update lr

                # measure elapsed time 迭代次信息输出
                batch_time.update(time.time() - iter_end)
                iter_end = time.time()
                if self.iteration % self.print_freq == 0:
                    log_str = 'Train: [{0}/{1}]\t epoch: {epoch:}\t iter: {iteration:}\t' \
                              'Time: {batch_time.val:.3f}  Data: {data_time.val:.3f}\t' \
                              'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                              'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t' \
                              'Prec@5: {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.6f}'.format(
                        batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                        lr=self.optim.param_groups[0]['lr'],
                        batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5)
                    print(log_str)
                    self.print_log(log_str)

            is_best = top1.avg > self.best_top1
            self.best_top1 = max(top1.avg, self.best_top1)
            self.best_top5 = max(top5.avg, self.best_top5)
            # 轮次信息输出
            epoch_time.update(time.time() - epoch_end)
            log_str = '\n Train_summary: [{0}/{1}/{top1.count:}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Epoch Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f}) Loss: {loss.avg:.4f}\t' \
                      'Prec@1: {top1.avg:.3f}  Prec@5: {top5.avg:.3f}  BestPrec@1:{best_top1:.3F} \t'\
                      'lr {lr:.6f}'.format(
                batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                lr=self.optim.param_groups[0]['lr'], epoch_time=epoch_time, data_time=data_time,
                loss=losses, top1=top1, top5=top5, best_top1=self.best_top1)

            print(log_str)
            self.print_log(log_str)

            # 保存模型
            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pth.tar')
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_top1': self.best_top1,
                'batch_time': batch_time,
                'losses': losses,
                'top1': top1,
                'top5': top5,
            }, checkpoint_file)
            if is_best:
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'model_best-{}.pth.tar'.format(self.timestamp)))
            if (self.epoch) % 500 == 0 and self.epoch !=1:  # save each 10 epoch
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'checkpoint-{}-{}.pth.tar'.format(self.epoch, time.strftime("%Y-%m-%d,%H,%M"))))

            # if self.iteration >= self.max_iter:
            #     break





