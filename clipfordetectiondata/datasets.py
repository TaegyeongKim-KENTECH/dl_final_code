# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the

import os
import random

import kornia.augmentation as K
import torch
from kornia.augmentation import RandomGaussianBlur, RandomJPEG
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from clipfordetectiondata.patchselect import patch_img

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

ImageFile.LOAD_TRUNCATED_IMAGES = True

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1),
)
jpeg_95 = RandomJPEG(jpeg_quality=(95, 95), p=1.0)
jpeg_90 = RandomJPEG(jpeg_quality=(90, 90), p=1.0)
jpeg_75 = RandomJPEG(jpeg_quality=(75, 75), p=1.0)
jpeg_50 = RandomJPEG(jpeg_quality=(50, 50), p=1.0)
blur_1_0 = RandomGaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0), p=1.0)
blur_2_0 = RandomGaussianBlur(kernel_size=(3, 3), sigma=(2.0, 2.0), p=1.0)
blur_3_0 = RandomGaussianBlur(kernel_size=(3, 3), sigma=(3.0, 3.0), p=1.0)

def safe_perturbations(x):
    try:
        return Perturbations(x)
    except RuntimeError as e:
        print(f"Error applying perturbations: {e}")
        return x

transform_before = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: safe_perturbations(x.unsqueeze(0)).squeeze(0)),
])
transform_before1 = transforms.Compose([
    transforms.ToTensor(),
])
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
])
transform_after_read = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
])
transform_train = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_before_test_jpeg95 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_95(x.unsqueeze(0)).squeeze(0)),
])
transform_before_test_jpeg90 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_90(x.unsqueeze(0)).squeeze(0)),
])
transform_before_test_jpeg75 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_75(x.unsqueeze(0)).squeeze(0)),
])
transform_before_test_jpeg50 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: jpeg_50(x.unsqueeze(0)).squeeze(0)),
])
transform_before_test_blur1_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: blur_1_0(x.unsqueeze(0)).squeeze(0)),
])
transform_before_test_blur2_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: blur_2_0(x.unsqueeze(0)).squeeze(0)),
])
transform_before_test_blur3_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: blur_3_0(x.unsqueeze(0)).squeeze(0)),
])

class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        root = args['data_path'] if is_train else args['eval_data_path']
        self.data_list = []

        if 'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root
            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):
                    assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append({
                            "image_path": os.path.join(file_path, folder_name, '0_real', image_path),
                            "label": 0,
                        })
                    for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                        self.data_list.append({
                            "image_path": os.path.join(file_path, folder_name, '1_fake', image_path),
                            "label": 1,
                        })
            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label": 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label": 1})
        else:
            for filename in os.listdir(root):
                file_path = os.path.join(root, filename)
                if '0_real' not in os.listdir(file_path):
                    for folder_name in os.listdir(file_path):
                        assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                        for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                            self.data_list.append({
                                "image_path": os.path.join(file_path, folder_name, '0_real', image_path),
                                "label": 0,
                            })
                        for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                            self.data_list.append({
                                "image_path": os.path.join(file_path, folder_name, '1_fake', image_path),
                                "label": 1,
                            })
                else:
                    for image_path in os.listdir(os.path.join(file_path, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label": 0})
                    for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label": 1})

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        try:
            x_min, x_max = patch_img(image)
        except Exception:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        image = transform_before(image)
        x_min = transform_before(x_min)
        x_max = transform_before(x_max)

        x_0 = transform_train(image)
        x_max = transform_train(x_max)
        x_min = transform_train(x_min)

        return torch.stack([x_max, x_min, x_0], dim=0), torch.tensor(int(targets))

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        root = args['data_path'] if is_train else args['eval_data_path']
        self.data_list = []
        file_path = root

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
                assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append({
                        "image_path": os.path.join(file_path, folder_name, '0_real', image_path),
                        "label": 0,
                        "folder": folder_name,
                    })
                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append({
                        "image_path": os.path.join(file_path, folder_name, '1_fake', image_path),
                        "label": 1,
                        "folder": folder_name,
                    })
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({
                    "image_path": os.path.join(file_path, '0_real', image_path),
                    "label": 0,
                    "folder": '0_real',
                })
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({
                    "image_path": os.path.join(file_path, '1_fake', image_path),
                    "label": 1,
                    "folder": '1_fake',
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, targets, folder = sample['image_path'], sample['label'], sample['folder']

        image = Image.open(image_path).convert('RGB')
        x_min, x_max = patch_img(image)
        image = transform_before_test(image)
        x_min = transform_before(x_min)
        x_max = transform_before(x_max)

        x_0 = transform_train(image)
        x_max = transform_train(x_max)
        x_min = transform_train(x_min)

        return torch.stack([x_max, x_min, x_0], dim=0), torch.tensor(int(targets)), folder

class TestDataset1(Dataset):
    def __init__(self, is_train, args):
        is_train = False
        root = args['data_path'] if is_train else args['eval_data_path']
        self.data_list = []
        self.folder_names = []
        file_path = root

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
                current_dir = os.path.join(file_path, folder_name)
                if not os.path.isdir(current_dir):
                    continue

                self.folder_names.append(folder_name)
                contents = sorted(os.listdir(current_dir))

                if '0_real' in contents and '1_fake' in contents:
                    for image_path in os.listdir(os.path.join(current_dir, '0_real')):
                        self.data_list.append({
                            "image_path": os.path.join(current_dir, '0_real', image_path),
                            "label": 0,
                            "folder": folder_name,
                        })
                    for image_path in os.listdir(os.path.join(current_dir, '1_fake')):
                        self.data_list.append({
                            "image_path": os.path.join(current_dir, '1_fake', image_path),
                            "label": 1,
                            "folder": folder_name,
                        })
                else:
                    for sub in contents:
                        sub_path = os.path.join(current_dir, sub)
                        if os.path.isdir(sub_path) and '0_real' in os.listdir(sub_path):
                            for image_path in os.listdir(os.path.join(sub_path, '0_real')):
                                self.data_list.append({
                                    "image_path": os.path.join(sub_path, '0_real', image_path),
                                    "label": 0,
                                    "folder": folder_name,
                                })
                            for image_path in os.listdir(os.path.join(sub_path, '1_fake')):
                                self.data_list.append({
                                    "image_path": os.path.join(sub_path, '1_fake', image_path),
                                    "label": 1,
                                    "folder": folder_name,
                                })
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({
                    "image_path": os.path.join(file_path, '0_real', image_path),
                    "label": 0,
                    "folder": '',
                })
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({
                    "image_path": os.path.join(file_path, '1_fake', image_path),
                    "label": 1,
                    "folder": '',
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, targets, folder = sample['image_path'], sample['label'], sample['folder']

        image = Image.open(image_path).convert('RGB')
        x_min, x_max = patch_img(image)
        image = transform_before_test(image)
        x_min = transform_before_test(x_min)
        x_max = transform_before_test(x_max)

        x_0 = transform_train(image)
        x_max = transform_train(x_max)
        x_min = transform_train(x_min)

        return torch.stack([x_max, x_min, x_0], dim=0), torch.tensor(int(targets)), folder
