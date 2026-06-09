import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageEnhance


def cv_random_flip(img_A, img_B, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img_A, img_B, label


def randomCrop_Mosaic(image_A, image_B, label, crop_win_width, crop_win_height):
    image_width = image_A.size[0]
    image_height = image_A.size[1]
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image_A.crop(random_region), image_B.crop(random_region), label.crop(random_region)


def randomRotation(image_A, image_B, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image_A = image_A.rotate(random_angle, mode)
        image_B = image_B.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image_A, image_B, label


def colorEnhance(image_A, image_B):
    bright_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Brightness(image_A).enhance(bright_intensity)
    image_B = ImageEnhance.Brightness(image_B).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Contrast(image_A).enhance(contrast_intensity)
    image_B = ImageEnhance.Contrast(image_B).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image_A = ImageEnhance.Color(image_A).enhance(color_intensity)
    image_B = ImageEnhance.Color(image_B).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image_A = ImageEnhance.Sharpness(image_A).enhance(sharp_intensity)
    image_B = ImageEnhance.Sharpness(image_B).enhance(sharp_intensity)
    return image_A, image_B


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training

class ChangeDataset(data.Dataset):
    def __init__(self, root, trainsize, mosaic_ratio=0.75):
        self.trainsize = trainsize
        self.mosaic_ratio = mosaic_ratio
        self.image_root_A = os.path.join(root, 'A/')
        self.image_root_B = os.path.join(root, 'B/')
        self.gt_root = os.path.join(root, 'label/')

        self.images_A_paths = sorted(
            [self.image_root_A + f for f in os.listdir(self.image_root_A) if f.endswith(('.jpg', '.png'))])
        self.images_B_paths = sorted(
            [self.image_root_B + f for f in os.listdir(self.image_root_B) if f.endswith(('.jpg', '.png'))])
        self.gts_paths = sorted([self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith(('.jpg', '.png'))])

        self.mode = 'direct'  
        self.tile_map = []
        with Image.open(self.images_A_paths[0]) as sample_img:
            width, height = sample_img.size

        if width > self.trainsize or height > self.trainsize:
            self.mode = 'tile'
            tiles_per_row = width // self.trainsize
            tiles_per_col = height // self.trainsize
            self.num_tiles = tiles_per_row * tiles_per_col

            for img_idx in range(len(self.images_A_paths)):
                for row in range(tiles_per_col):
                    for col in range(tiles_per_row):
                        self.tile_map.append((img_idx, row, col))
            self.size = len(self.tile_map)
        else:
            self.size = len(self.images_A_paths)

    
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.gt_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        if self.mode == 'tile':
            img_idx, row, col = self.tile_map[index]
            top = row * self.trainsize
            left = col * self.trainsize
            box = (left, top, left + self.trainsize, top + self.trainsize)

            image_A_large = Image.open(self.images_A_paths[img_idx]).convert('RGB')
            image_B_large = Image.open(self.images_B_paths[img_idx]).convert('RGB')
            gt_large = Image.open(self.gts_paths[img_idx]).convert('L')

            image_A = image_A_large.crop(box)
            image_B = image_B_large.crop(box)
            gt = gt_large.crop(box)
        else:
            image_A = Image.open(self.images_A_paths[index]).convert('RGB')
            image_B = Image.open(self.images_B_paths[index]).convert('RGB')
            gt = Image.open(self.gts_paths[index]).convert('L')

        image_A, image_B, gt = cv_random_flip(image_A, image_B, gt)
        image_A, image_B, gt = randomRotation(image_A, image_B, gt)
        image_A, image_B = colorEnhance(image_A, image_B)
        gt = randomPeper(gt)

        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        gt = self.gt_transform(gt)

        return image_A, image_B, gt

    def __len__(self):
        return self.size

def get_loader(root, batchsize, trainsize, num_workers=1, shuffle=True, pin_memory=True):

    dataset =ChangeDataset(root = root, trainsize= trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class Test_ChangeDataset(data.Dataset):
    def __init__(self, root, testsize):
        self.testsize = testsize
        self.image_root_A = os.path.join(root, 'A/')
        self.image_root_B = os.path.join(root, 'B/')
        self.gt_root = os.path.join(root, 'label/')
        self.images_A_paths = sorted(
            [self.image_root_A + f for f in os.listdir(self.image_root_A) if f.endswith(('.jpg', '.png'))])
        self.images_B_paths = sorted(
            [self.image_root_B + f for f in os.listdir(self.image_root_B) if f.endswith(('.jpg', '.png'))])
        self.gts_paths = sorted([self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith(('.jpg', '.png'))])
        self.mode = 'direct'
        self.tile_map = []
        self.large_image_info = {}  
        with Image.open(self.images_A_paths[0]) as sample_img:
            width, height = sample_img.size
        if width > self.testsize or height > self.testsize:
            self.mode = 'tile'
            tiles_per_row = (width + self.testsize - 1) // self.testsize  
            tiles_per_col = (height + self.testsize - 1) // self.testsize  
            for img_idx, img_path in enumerate(self.images_A_paths):
                base_name = os.path.basename(img_path)
                with Image.open(img_path) as img:
                    w, h = img.size
                    w_tiles = (w + self.testsize - 1) // self.testsize
                    h_tiles = (h + self.testsize - 1) // self.testsize
                    self.large_image_info[base_name] = {'width': w, 'height': h, 'tiles_w': w_tiles, 'tiles_h': h_tiles}
                for row in range(h_tiles):
                    for col in range(w_tiles):
                        self.tile_map.append((img_idx, row, col))
            self.size = len(self.tile_map)
        else:
            self.size = len(self.images_A_paths)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.gt_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        if self.mode == 'tile':
            img_idx, row, col = self.tile_map[index]

            large_A = Image.open(self.images_A_paths[img_idx]).convert('RGB')
            large_B = Image.open(self.images_B_paths[img_idx]).convert('RGB')
            large_gt = Image.open(self.gts_paths[img_idx]).convert('L')

            w, h = large_A.size

            left = col * self.testsize
            top = row * self.testsize
            right = min(left + self.testsize, w)
            bottom = min(top + self.testsize, h)

            tile_A = large_A.crop((left, top, right, bottom))
            tile_B = large_B.crop((left, top, right, bottom))
            tile_gt = large_gt.crop((left, top, right, bottom))

            if tile_A.size != (self.testsize, self.testsize):
                padded_A = Image.new('RGB', (self.testsize, self.testsize), (0, 0, 0))
                padded_B = Image.new('RGB', (self.testsize, self.testsize), (0, 0, 0))
                padded_gt = Image.new('L', (self.testsize, self.testsize), 0)
                padded_A.paste(tile_A, (0, 0))
                padded_B.paste(tile_B, (0, 0))
                padded_gt.paste(tile_gt, (0, 0))
                tile_A, tile_B, tile_gt = padded_A, padded_B, padded_gt

            img_A = self.img_transform(tile_A)
            img_B = self.img_transform(tile_B)
            gt = self.gt_transform(tile_gt)

            base_name = os.path.basename(self.images_A_paths[img_idx])
            info = self.large_image_info[base_name].copy()
            info.update({'row': row, 'col': col})

            return img_A, img_B, gt, base_name, info

        else: 
            img_A = Image.open(self.images_A_paths[index]).convert('RGB')
            img_B = Image.open(self.images_B_paths[index]).convert('RGB')
            gt = Image.open(self.gts_paths[index]).convert('L')

            base_name = os.path.basename(self.images_A_paths[index])
            w, h = img_A.size

            img_A = self.img_transform(img_A)
            img_B = self.img_transform(img_B)
            gt = self.gt_transform(gt)

            info = {'width': w, 'height': h, 'tiles_w': 1, 'tiles_h': 1, 'row': 0, 'col': 0}

            return img_A, img_B, gt, base_name, info

    def __len__(self):
        return self.size

def get_test_loader(root, batchsize, testsize, num_workers=1, shuffle=True, pin_memory=True):

    dataset =Test_ChangeDataset(root = root, testsize=testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


