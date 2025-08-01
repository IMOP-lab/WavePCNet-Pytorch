import os
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

def get_image_list(name, config, phase):
    images = []
    gts = []

    if name == 'all':
        print("Skipping 'all' dataset as it does not have a corresponding directory.")
        return images, gts

    if name in ('simple', 'tough', 'normal'):
        train_split = 10000
        
        print('Objectness shifting experiment.')
        # Objectness
        list_file = 'clean_list.txt'
        f = open(os.path.join(config['data_path'], 'SALOD/{}'.format(list_file)), 'r')
        if name == 'simple':
            img_list = f.readlines()[-train_split:]
        elif name == 'normal':
            img_list = f.readlines()[train_split:-train_split]
        else:
            img_list = f.readlines()[:train_split]
                
        for i in range(len(img_list)):
            img_list[i] = img_list[i].split(' ')[0]
                
        images = [os.path.join(config['data_path'], 'SALOD/images', line.strip() + '.jpg') for line in img_list]
        gts = [os.path.join(config['data_path'], 'SALOD/mask', line.strip() + '.png') for line in img_list]
        
    # Benchmark
    elif name == 'SALOD':
        f = open(os.path.join(config['data_path'], 'SALOD/{}.txt'.format(phase)), 'r')
        img_list = f.readlines()
        
        images = [os.path.join(config['data_path'], name, 'images', line.strip() + '.jpg') for line in img_list]
        gts = [os.path.join(config['data_path'], name, 'mask', line.strip() + '.png') for line in img_list]

    elif phase == 'test' and os.path.isabs(name):
        image_root = os.path.join(name, 'images')
        gt_root = os.path.join(name, 'segmentations')
        
        images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])       

    else:
        image_root = os.path.join(config['data_path'], name, 'images')
        print("Name:", name)
        print("Image root:", image_root)
        gt_root = os.path.join(config['data_path'], name, 'segmentations')
        
        images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
    
    return images, gts

def get_loader(config):
    dataset = Train_Dataset(config['trset'], config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=12,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader



class Train_Dataset(data.Dataset):
    def __init__(self, name, config):
        self.config = config
        self.images, self.gts = get_image_list(name, config, 'train')
        self.size = len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        
        img_size = self.config['size']
        image = image.resize((img_size, img_size))
        gt = gt.resize((img_size, img_size))
    
        image = np.array(image).astype(np.float32)
        gt = np.array(gt)
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        gt = np.expand_dims((gt > 128).astype(np.float32), axis=0)

        image = torch.from_numpy(image).float()
        gt = torch.from_numpy(gt).float()
        return image, gt

    def __len__(self):
        return self.size

class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        self.images, self.gts = get_image_list(name, config, 'test')
        self.size = len(self.images)

    def load_data(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = image.resize((self.config['size'], self.config['size']))
        image = np.array(image).astype(np.float32)
        gt = np.array(Image.open(self.gts[index]).convert('L'))
        gt = Image.open(self.gts[index]).convert('L')
        gt = gt.resize((self.config['size'], self.config['size']))
        gt = np.array(gt).astype(np.float32)
        name = self.images[index].split('/')[-1].split('.')[0]
        
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = torch.tensor(np.expand_dims(image, 0)).float()
        gt = (gt > 128).astype(np.float32)
        return image, gt, name

def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    data_loader = Test_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))
    

if __name__ == "__main__":
    test_data()