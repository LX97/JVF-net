import os
import numpy as np
import torch
import librosa
import random
import wave
import librosa
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
from scipy.io import wavfile

Transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def load_face(face_path):
    # NOTE: 3 channels are in BGR order
    img = Image.open(face_path)
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    if img.layers != 3 or img.size != (224, 224):    # 灰度图转为彩图
        img = img.convert('RGB').resize((224, 224), resample=Image.BILINEAR)

    img = Transform(img)
    return img


class VGG_Face_Dataset(Dataset):
    def __init__(self, face_voice_dir, mode, load_raw=False):
        face_voice_list = np.load(face_voice_dir, allow_pickle=True)
        self.face_voice_list = face_voice_list.tolist()
        self.speakers_num = len(self.face_voice_list)  # 计算发言者数量

    def __getitem__(self, index):
        positive = self.face_voice_list[index]
        label = positive['id_num']
        real_face_path = positive['face_path'][np.random.randint(0, len(positive['face_path']))]
        real_face = load_face(real_face_path)

        return real_face, label

    def __len__(self):
        return len(self.face_voice_list)


class Dataset(Dataset):
    def __init__(self, data_dir, mode, fixed_offset, load_raw=False):
        # self.data_dir = data_dir
        self.fixed_offset = fixed_offset
        self.load_raw = load_raw
        all_triplets = np.load(data_dir, allow_pickle=True)
        self.all_triplets = all_triplets.tolist()
        self.speakers_num = len(self.all_triplets)  # 计算发言者数量


    def __getitem__(self, p_index):
        n_index = p_index

        positive = self.all_triplets[p_index]

        while(n_index == p_index):
            n_index = np.random.randint(0, self.speakers_num)   # 计算 0~1225之间的随机数

        negative = self.all_triplets[n_index]

        real_audio_path = positive['voice_path'][np.random.randint(0, len(positive['voice_path']))]
        real_face_path =  positive['face_path'][np.random.randint(0, len(positive['face_path']))]
        fake_face_path = negative['face_path'][np.random.randint(0, len(negative['face_path']))]

        real_audio = self.load_audio(real_audio_path)
        real_face = load_face(real_face_path)
        fake_face = load_face(fake_face_path)
        which_side = random.randint(0, 1)
        if which_side == 0:
            ground_truth = torch.LongTensor([0])
            face_a = real_face
            face_b = fake_face
        else:
            ground_truth = torch.LongTensor([1])
            face_a = fake_face
            face_b = real_face
        return real_audio, face_a, face_b, ground_truth

    def load_audio(self, audio_path):
        y = librosa.load(audio_path)
        y = y[0]
        if self.fixed_offset:
            offset = 0
        else:
            max_offset = y.shape[0] - 48000
            offset = random.randint(0, max_offset)
        y = y[offset:offset+48000]
        if self.load_raw:
            y = np.expand_dims(y, axis=0)
            return y
        spect = Dataset.get_spectrogram(y)
        for i in range(spect.shape[1]):
            f_bin = spect[:, i]
            f_bin_mean = np.mean(f_bin)
            f_bin_std = np.std(f_bin)
            spect[:, i] = (spect[:, i] - f_bin_mean) / (f_bin_std + 1e-7)
        spect = np.expand_dims(spect, axis=0)
        return spect

    def __len__(self):
        return len(self.all_triplets)

    @staticmethod
    def get_spectrogram(y, n_fft=1024, hop_length=160, win_length=400, window='hamming'):
        y_hat = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        y_hat = y_hat[:-1, :-1]
        D = np.abs(y_hat)
        return D


def custom_collate_fn(batch):
    real_audio = [torch.from_numpy(item[0]) for item in batch]
    face_a = [item[1] for item in batch]
    face_b = [item[2] for item in batch]
    gt = [item[3] for item in batch]
    real_audio = torch.stack(real_audio, dim=0)
    face_a = torch.stack(face_a, dim=0)
    face_b = torch.stack(face_b, dim=0)
    gt = torch.cat(gt, dim=0)
    return [real_audio, face_a, face_b, gt]


if __name__ == '__main__':

    # dataset = Dataset( './dataset/voclexb-VGG_face-datasets/voice_face_list.npy', 'train', False)
    # loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=8, collate_fn=custom_collate_fn)
    #
    # for step, (real_audio, face_a, face_b, ground_truth) in enumerate(loader):
    #     print(real_audio.shape)  # (B, 1, 512, 300)
    #     print(face_a.shape)  # (B, 3, 224, 224)
    #     print(face_b.shape)
    #     print(ground_truth.shape)  # (B)

    face_dataset = VGG_Face_Dataset('./dataset/voclexb-VGG_face-datasets/voice_face_list.npy', 'train')
    face_loader = DataLoader(face_dataset, batch_size=24, shuffle=True, drop_last=False, num_workers=8)

    for step, (real_face, ground_truth) in enumerate(face_loader):
        print(real_face.shape)  # (B, 1, 512, 300)
        print(ground_truth.shape)  # (B)
        print(step)