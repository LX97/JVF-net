# coding=utf-8
import time
import torch
import torch.nn as nn
import model.model4 as model4
from torch.utils.data import DataLoader
from dataset import VGG_Face_Dataset
from model.model4 import ResNet
# import models.resnet as ResNet
# import models.senet as SENet
# import models.vgg as VGG


# from face_image_embeddding.utils import utils, face_dataset
from trainer import Trainer


configurations = {
    'network': dict(
        type='resnet',
        class_num = 24),

    'training': dict(
        start_epoch=0,
        start_iteration = 0,
        batch_size = 64,
        max_epoch= 3000,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        gamma=0.9,  # "lr_policy: step"
        step_size=1000,  # "lr_policy: step"
        interval_validate=1000,
    ),
}



def main():
    timestamp = time.strftime("%Y-%m-%d,%H,%M")
    attribution = "identity"     # identity, emotion 类型

    pre_weight_file = '../weight/resnet/resnet50_scratch_weight.pkl'
    log_file = '../log/log_file_{}.log'.format(timestamp)
    checkpoint_dir = '../saved'
    include_top = True #

    train_cfg = configurations['training']
    net_cfg = configurations['network']
    cuda = torch.cuda.is_available()


    face_dataset = VGG_Face_Dataset('../dataset/voclexb-VGG_face-datasets/voice_face_list.npy', 'train')
    face_loader = DataLoader(face_dataset, batch_size=24, drop_last=False,
                             shuffle=True, num_workers=8, pin_memory=True)

    val_loader = None

    if net_cfg['type'] == 'resnet':
        model = ResNet(num_classes=face_dataset.speakers_num, include_top=include_top)
        # utils.load_state_dict(model, weight_file)
        # model.fc.reset_parameters()

    criterion = nn.CrossEntropyLoss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optim = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg['lr'],
        momentum=train_cfg['momentum'],
        weight_decay=train_cfg['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, train_cfg['step_size'], gamma=train_cfg['gamma'], last_epoch=-1)
    trainer = Trainer(
        cmd='train',
        cuda=cuda,
        model=model,
        criterion=criterion,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        train_loader=face_loader,
        val_loader=val_loader,
        log_file=log_file,
        max_epoch=train_cfg['max_epoch'],
        timestamp=timestamp,
        checkpoint_dir=checkpoint_dir,
        print_freq=1,
    )
    trainer.print_log(str(net_cfg))
    trainer.print_log(str(train_cfg))
    trainer.epoch = 0
    trainer.iteration = 0
    trainer.train()



if __name__ == '__main__':
    main()