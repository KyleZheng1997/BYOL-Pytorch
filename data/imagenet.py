from random import sample
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
import mc
import io
import random
import torch
import ctypes
import multiprocessing as mp
import numpy as np


class DatasetCache(data.Dataset):
    def __init__(self):
        super().__init__()
        self.initialized = False
    

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def load_image(self, filename):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        
        buff = io.BytesIO(value_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img



class BaseDataset(DatasetCache):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__()
        self.initialized = False


        prefix = '/mnt/lustre/share/images/meta'
        image_folder_prefix = '/mnt/lustre/share/images'
        if mode == 'train':
            image_list = os.path.join(prefix, 'train.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'train')
        elif mode == 'test':
            image_list = os.path.join(prefix, 'test.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'test')
        elif mode == 'val':
            image_list = os.path.join(prefix, 'val.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'val')
        else:
            raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, eval]')


        self.samples = []
        with open(image_list) as f:
            for line in f:
                name, label = line.split()
                label = int(label)
                if label < max_class:
                    self.samples.append((label, name))

        if aug is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = aug




class ImagenetContrastive(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        _, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        if isinstance(self.transform, list):
            return self.transform[0](img), self.transform[1](img)
        return self.transform(img), self.transform(img)



class ImagenetContrastiveWithLabel(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), self.transform(img), label


class Imagenet(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), label


class ImagenetPercent(DatasetCache):
    def __init__(self, percent, aug=None):
        super().__init__()
        classes = [d.name for d in os.scandir('/mnt/lustre/share/images/train') if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        if percent == 1:
            image_list = 'semi_files/1percent.txt'
        elif percent == 10:
            image_list = 'semi_files/10percent.txt'
        else:
            raise NotImplementedError('you have to choose from 1 percent or 10 percent')

        self.samples = []
        with open(image_list) as f:
            for line in f:
                name = line.strip()
                class_name = name.split('_')[0]
                label = class_to_idx[class_name]
                name = class_name + '/' + name
                self.samples.append((label, name))

        if aug is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = aug

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join('/mnt/lustre/share/images/train/', name)
        img = self.load_image(filename)
        return self.transform(img), label




class ImagenetContrastiveWithIndex(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None, topk=1):
        super().__init__(mode, max_class, aug)

        num_images = self.samples.__len__()
        shared_array_base = mp.Array(ctypes.c_long, num_images * topk)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(num_images, topk)
        self.knn = shared_array

    def update_knn(self, indices, similar_indices):
        self.knn[indices] = similar_indices

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        _, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)

        similar = []
        if not self.knn is None:
            for i in self.knn[index]:
                _, name = self.samples[i]
                filename = os.path.join(self.image_folder, name)
                similar_img = self.load_image(filename)
                similar.append(self.transform(similar_img))
            similar = torch.stack(similar)
        return self.transform(img), self.transform(img), index, similar






class ImagenetSupContrast(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

        class_dict = {}
        for (label, name) in self.samples:
            if label in class_dict:
                class_dict[label].append(name)
            else:
                class_dict[label] = [name]
        self.class_dict = class_dict

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        
        pos_name = random.choice(self.class_dict[label])
        pos_filename = os.path.join(self.image_folder, pos_name)
        pos_img = self.load_image(pos_filename)
        
        return self.transform(img), self.transform(pos_img), label

