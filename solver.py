import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from dataset import custom_collate_fn
import importlib
import os


class Solver():
    def __init__(self, config):
        def weight_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if config['weight_init'] == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif config['weight_init'] == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif config['weight_init'] == 'gaussian':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    pass
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # load_raw = True if config['model'] == 'model4' or config['model'] == 'model5' else False
        load_raw =False
        self.train_data = Dataset(data_dir=config['data_dir'],
                                  mode='train',
                                  fixed_offset=False,
                                  load_raw=load_raw)
        self.val_data = Dataset(data_dir=config['data_dir'],
                                mode='val',
                                fixed_offset=True,
                                load_raw=load_raw)
        self.test_data = Dataset(data_dir=config['data_dir'],
                                 mode='test',
                                 fixed_offset=True,
                                 load_raw=load_raw)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       shuffle=True,
                                       drop_last=True,
                                       collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(self.val_data,
                                     batch_size=config['batch_size'],
                                     num_workers=config['num_workers'],
                                     shuffle=True,
                                     drop_last=True,
                                     collate_fn=custom_collate_fn)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=config['batch_size'],
                                      num_workers=config['num_workers'],
                                      shuffle=False,
                                      drop_last=True,
                                      collate_fn=custom_collate_fn)
        model = importlib.import_module('model.{}'.format(config['model']))
        if config['model'] != 'model4' and config['model'] != 'model5':
            self.net = model.SVHFNet().to(self.device)
            if config['weight_init'] == 'xavier_uniform':
                print('Initialize weight with xavier_uniform.')
            elif config['weight_init'] == 'kaiming_uniform':
                print('Initialize weight with kaiming_uniform.')
            elif config['weight_init'] == 'gaussian':
                print('Initialize weight with gaussian.')
            else:
                print('Initialize weight with defualt setting.')
            self.net.apply(weight_init)
        else:
            self.net = model.SVHFNet(config['res_ckpt_path'],
                                     config['pase_cfg_path'],
                                     config['pase_ckpt_path']).to(self.device)
            print('Don\'t apply any weight init method for model4.')
        if config['load_model']:
            print('Load pretrained model:', config['load_path'])
            checkpoint = torch.load(config['load_path'])
            state_dict = checkpoint['net']
            self.net.load_state_dict(state_dict)

        if config['multi_gpu']:
            print('Use Multi GPU.')
            self.net = nn.DataParallel(self.net, device_ids=config['gpu_ids'])

        self.criterion = nn.CrossEntropyLoss()
        if config['optim'] == 'sgd':
            print('Use SGD optimizer.')
            self.optim = torch.optim.SGD(params=self.net.parameters(),
                                         lr=config['lr'],
                                         momentum=0.9,
                                         weight_decay=0.0005)
        else:
            print('Use Adam optimizer.')
            self.optim = torch.optim.Adam(params=self.net.parameters(),
                                          lr=config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim,
                                                                    factor=config['lr_decay_factor'],
                                                                    patience=config['patience'],
                                                                    verbose=True)
        self.saved_dir = os.path.join(config['save_dir'], config['model_name'])

    def fit(self):
        print('Start training..')
        for epoch in range(self.config['epoch']):
            for step, (real_audio, face_a, face_b, labels) in enumerate(self.train_loader):
                real_audio = real_audio.to(self.device)
                face_a = face_a.to(self.device)
                face_b = face_b.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(face_a, face_b, real_audio)
                loss = self.criterion(outputs, labels)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                _, argmax = torch.max(outputs, 1)
                accuracy = (labels == argmax.squeeze()).float().mean()

                print('Epoch[{}/{}]  Step[{}/{}]  Loss: {:.8f}  Accuracy: {:.2f}%'.format(
                    epoch + 1, self.config['epoch'], step + 1,
                    self.train_data.__len__() // self.config['batch_size'],
                    loss.item(), accuracy.item() * 100
                ))

            if (epoch + 1) % self.config['val_every'] == 0:
                val_loss = self.val(epoch + 1)
                self.scheduler.step(val_loss)
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save(epoch + 1)

    def val(self, epoch):
        print('Start validation..')
        self.net.eval()
        cnt = 0
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for step, (real_audio, face_a, face_b, labels) in enumerate(self.val_loader):
                real_audio = real_audio.to(self.device)
                face_a = face_a.to(self.device)
                face_b = face_b.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(face_a, face_b, real_audio)

                loss = self.criterion(outputs, labels)
                total_loss += loss
                _, argmax = torch.max(outputs, 1)
                accuracy = (labels == argmax.squeeze()).float().mean()
                total_acc += accuracy

                print('[val] Step[{}/{}]  Loss: {:.8f}  Accuracy: {:.2f}%'.format(
                    step + 1,
                    self.val_data.__len__() // self.config['batch_size'],
                    loss.item(), accuracy.item() * 100
                ))
                cnt += 1
            average_loss = total_loss / cnt
            average_acc = total_acc / cnt
            print('[Validation {}]  Average Loss: {:.8f}  Average Accuracy: {:.2f}'.format(
                epoch, average_loss, average_acc))
        self.net.train()
        return average_loss

    def test(self):
        print('Start test..')
        self.net.eval()
        cnt = 0
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for step, (real_audio, face_a, face_b, labels) in enumerate(self.test_loader):
                real_audio = real_audio.to(self.device)
                face_a = face_a.to(self.device)
                face_b = face_b.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(face_a, face_b, real_audio)
                loss = self.criterion(outputs, labels)
                total_loss += loss
                _, argmax = torch.max(outputs, 1)
                accuracy = (labels == argmax.squeeze()).float().mean()
                total_acc += accuracy

                print('[test] Step[{}/{}]  Loss: {:.8f}  Accuracy: {:.2f}%'.format(
                    step + 1,
                    self.test_data.__len__() // self.config['batch_size'],
                    loss.item(), accuracy.item() * 100
                ))
                cnt += 1
            average_loss = total_loss / cnt
            average_acc = total_acc / cnt
            print('[Test]  Average Loss: {:.8f}  Average Accuracy: {:.2f}'.format(
                average_loss, average_acc))
        self.net.train()

    def save(self, epoch):
        os.makedirs(self.saved_dir, exist_ok=True)
        if self.config['multi_gpu']:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()

        checkpoint = {
            'net': state_dict
        }

        output_path = os.path.join(self.saved_dir, 'model_' + str(epoch) + '.pt')
        torch.save(checkpoint, output_path)
