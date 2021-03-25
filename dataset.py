import os
import numpy as np
import torch
import librosa
import random
import wave
import librosa
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import wavfile

class VGG_Face_Dataset(Dataset):
    def __init__(self, face_list, mode, fixed_offset, load_raw=False):



class Dataset(Dataset):
    def __init__(self, data_dir, mode, fixed_offset, load_raw=False):
        # self.data_dir = data_dir
        self.fixed_offset = fixed_offset
        self.load_raw = load_raw

        all_triplets = np.load(data_dir, allow_pickle=True)
        self.all_triplets = all_triplets.tolist()

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __getitem__(self, p_index):
        n_index = p_index
        speakers_num = len(self.all_triplets)
        positive = self.all_triplets[p_index]

        while(n_index == p_index):
            n_index = np.random.randint(0, speakers_num)   #计算 0~1225之间的随机数

        negative = self.all_triplets[n_index]

        real_audio_path = positive['voice_path'][np.random.randint(0, len(positive['voice_path']))]
        real_face_path =  positive['face_path'][np.random.randint(0, len(positive['face_path']))]
        fake_face_path = negative['face_path'][np.random.randint(0, len(negative['face_path']))]

        real_audio = self.load_audio(real_audio_path)
        real_face = self.load_face(real_face_path)
        fake_face = self.load_face(fake_face_path)
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

    def load_face(self, face_path):
        # NOTE: 3 channels are in BGR order
        img = Image.open(face_path)
        if img.size != (224, 224):
            img = img.resize((224, 224), resample=Image.BILINEAR)
        img = self.transform(img)
        return img

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
    from torch.utils.data import DataLoader

    dataset = Dataset( './dataset/voclexb-VGG_face-datasets/voice_face_list.npy', 'train', False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=8, collate_fn=custom_collate_fn)

    for step, (real_audio, face_a, face_b, ground_truth) in enumerate(loader):
        print(real_audio.shape)  # (B, 1, 512, 300)
        print(face_a.shape)  # (B, 3, 224, 224)
        print(face_b.shape)
        print(ground_truth.shape)  # (B)
        print(ground_truth)

        break
