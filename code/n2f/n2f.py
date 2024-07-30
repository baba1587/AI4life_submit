"""
    code for AI4Life Training using Noise2xx methods
    Time: 12:20 20/07 2024
"""

import tifffile
import numpy as np
import argparse
import os, random, torch, time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as ts
from tqdm import tqdm
import torch.nn.functional as F

# enable anomaly detection
torch.autograd.set_detect_anomaly(True)

################################
# Utils
################################


def getTiffImage(image_path):
    with tifffile.TiffFile(image_path) as tif:
        images = tif.asarray()
        return images


def getImageInfo(image):
    print(image.shape)
    print(image.min())
    print(image.max())


# random seed definition
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def n2fTransform(img):
    """
        Pytorch method for n2f transform
    
    """
    listimgH = []
    shape = img.shape  #[-2:]
    Zshape = [shape[0], shape[1]]
    if shape[0] % 2 == 1:
        Zshape[0] -= 1
    if shape[1] % 2 == 1:
        Zshape[1] -= 1
    imgZ = img[:Zshape[0], :Zshape[1]]

    imgin = torch.zeros((Zshape[0] // 2, Zshape[1]), dtype=torch.float32)
    imgin2 = torch.zeros((Zshape[0] // 2, Zshape[1]), dtype=torch.float32)
    for i in range(imgin.shape[0]):
        for j in range(imgin.shape[1]):
            if j % 2 == 0:
                imgin[i, j] = imgZ[2 * i + 1, j]
                imgin2[i, j] = imgZ[2 * i, j]
            if j % 2 == 1:
                imgin[i, j] = imgZ[2 * i, j]
                imgin2[i, j] = imgZ[2 * i + 1, j]
    listimgH.append(imgin.unsqueeze(0))
    listimgH.append(imgin2.unsqueeze(0))

    listimgV = []
    Zshape = [shape[0], shape[1]]
    if shape[0] % 2 == 1:
        Zshape[0] -= 1
    if shape[1] % 2 == 1:
        Zshape[1] -= 1
    imgZ = img[:Zshape[0], :Zshape[1]]

    imgin3 = torch.zeros((Zshape[0], Zshape[1] // 2), dtype=torch.float32)
    imgin4 = torch.zeros((Zshape[0], Zshape[1] // 2), dtype=torch.float32)
    for i in range(imgin3.shape[0]):
        for j in range(imgin3.shape[1]):
            if i % 2 == 0:
                imgin3[i, j] = imgZ[i, 2 * j + 1]
                imgin4[i, j] = imgZ[i, 2 * j]
            if i % 2 == 1:
                imgin3[i, j] = imgZ[i, 2 * j]
                imgin4[i, j] = imgZ[i, 2 * j + 1]
    listimgV.append(imgin3.unsqueeze(0))
    listimgV.append(imgin4.unsqueeze(0))

    listimgV1 = [[listimgV[0], listimgV[1]]]
    listimgV2 = [[listimgV[1], listimgV[0]]]
    listimgH1 = [[listimgH[1], listimgH[0]]]
    listimgH2 = [[listimgH[0], listimgH[1]]]
    listimg = listimgH1 + listimgH2 + listimgV1 + listimgV2
    return listimg


def AI4LifeTransform(sample, args):
    def normalize(image, min_val=0.0, max_val=1.0):

        if isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.float32)
            image = torch.tensor(image, dtype=torch.float32)

        image_min = torch.min(image)
        image_max = torch.max(image)

        normalized_image = (image - image_min) / (image_max - image_min)
        normalized_image = normalized_image * (max_val - min_val) + min_val
        return normalized_image

    def train_transforms(sample):
        transforms = ts.Compose([
            ts.RandomCrop(args.train_size),
            ts.RandomHorizontalFlip(p=0.5),
            ts.RandomVerticalFlip(p=0.5),
        ])

        transformed_sample = transforms(sample)
        return n2fTransform(transformed_sample.squeeze())

    def test_transforms(sample):
        transforms = ts.Compose([])
        return transforms(sample)

    sample = normalize(sample, min_val=0.0, max_val=1.0)
    sample = sample.unsqueeze(0)

    if args.mode == 'train':
        sample = train_transforms(sample)

    if args.mode == 'test':
        sample = test_transforms(sample)

    return sample


################################
# dataset
################################
class AI4LifeDataset(Dataset):
    def __init__(self, dataset_path, args):
        super(AI4LifeDataset).__init__()
        self.dataset_path = dataset_path
        self.image = getTiffImage(self.dataset_path)
        self.images = self.preprocessImage()

        # image_num = len(self.images)
        # print(f"self.images len : {len(self.images)}, image_num : {image_num}")
        self.transform = args.mode
        self.args = args

    def __len__(self):
        # print(f"self.images.shape : {self.images.shape}")
        return len(self.images)

    def preprocessImage(self):
        """
        Extract all slices in the data array into 3D tensors.
        """
        train_image_list = []
        if len(self.image.shape) == 4:
            ia, ib, _, _ = self.image.shape
            for a in range(ia):
                for b in range(ib):
                    train_image_list.append(self.image[a, b, :, :])

        elif len(self.image.shape) == 3:
            ia, _, _ = self.image.shape
            for a in range(ia):
                train_image_list.append(self.image[a, :, :])
        else:
            print(f"No eligible information needs to be extracted !")

        images = np.array(train_image_list)
        # print(f"dataset has {len(images)} slices with the original shape {images.shape}")
        return images

    def __getitem__(self, index):
        train_image = self.images[index]
        train_image = AI4LifeTransform(train_image, args=self.args)

        return train_image


################################
# Network
################################


class TwoCon(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoCon(1, 64)
        self.conv2 = TwoCon(64, 64)
        self.conv3 = TwoCon(64, 64)
        self.conv4 = TwoCon(64, 64)
        self.conv6 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = torch.clamp(self.conv6(x4) + x, 0.0, 1.0)
        return x5


class MapNN_No_Tconv(nn.Module):
    """
        Red cnn generator
    """
    def __init__(self, out_ch=32):
        super(MapNN_No_Tconv, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(1,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(out_ch,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(out_ch,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.conv4 = nn.Conv2d(out_ch,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.conv5 = nn.Conv2d(out_ch,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)

        self.tconv1 = nn.Conv2d(out_ch,
                                out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True)
        self.tconv2 = nn.Conv2d(out_ch,
                                out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True)
        self.tconv3 = nn.Conv2d(out_ch,
                                out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True)
        self.tconv4 = nn.Conv2d(out_ch,
                                out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.tconv5 = nn.Conv2d(out_ch,
                                1,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)

        self.conv6 = nn.Conv2d(out_ch * 2,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.conv7 = nn.Conv2d(out_ch * 2,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.conv8 = nn.Conv2d(out_ch * 2,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.conv9 = nn.Conv2d(out_ch * 2,
                               out_ch,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight,
                                        gain=nn.init.calculate_gain('relu'))

    def forward(self, x):

        # encoder

        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out5 = self.relu(self.conv5(out4))

        # decoder
        out6 = self.relu(torch.cat([self.tconv1(out5), out4], dim=1))
        out6 = self.relu(self.conv6(out6))

        out7 = self.relu(torch.cat([self.tconv2(out6), out3], dim=1))
        out7 = self.relu(self.conv7(out7))

        out8 = self.relu(torch.cat([self.tconv3(out7), out2], dim=1))
        out8 = self.relu(self.conv8(out8))

        out9 = self.relu(torch.cat([self.tconv4(out8), out1], dim=1))
        out9 = self.relu(self.conv9(out9))

        out10 = self.relu(self.tconv5(out9) + x)

        out10 = torch.clamp(out10, 0.0, 1.0)

        return out10


################################
# args
################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_epoch", type=int, default=400)
    parser.add_argument("--train_batchsize", type=int, default=64)
    parser.add_argument("--numworkers", type=int, default=8)
    parser.add_argument("--data_root", type=str, default=" ")
    parser.add_argument("--save_path_root", type=str, default=" ")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # set up
    set_seed()

    DATA_ROOT = args.data_root

    path_u1 = DATA_ROOT + "/u1.tiff"
    path_u2 = DATA_ROOT + "/u2.tiff"
    path_s1 = DATA_ROOT + "/s1.tiff"
    path_s2 = DATA_ROOT + "/s2.tiff"

    dataset = AI4LifeDataset(dataset_path=path_s1, args=args)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.train_batchsize,
                            shuffle=True,
                            num_workers=args.numworkers,
                            pin_memory=True,
                            drop_last=True)

    train(dataloader=dataloader, args=args)
    print("training complete!")


def train(dataloader, args):
    total_loss = 0.0
    mse_loss = nn.BCELoss()  #.to(args.device)
    # net = MapNN_No_Tconv(out_ch=32).to(args.device)
    net = Net().to(args.device)

    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-05,
                                  weight_decay=0.)

    learning_schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=80,
        T_mult=2,
        eta_min=1e-8,
        last_epoch=-1,
        verbose=True)

    for epoch in range(1, args.train_epoch):
        start_time = time.time()
        net.train()
        print("training epoch : ", epoch)
        for batch_idx, (data) in enumerate(dataloader):
            index = np.random.randint(0, len(data))
            data_index = data[index]
            input = data_index[0].to(args.device)
            label = data_index[1].to(args.device)
            # print(f'Batch Index: {batch_idx}')
            # print(f'Data Shape: {input.shape} and label shape : {label.shape}')

            output = net(input)
            # assert output.shape == input.shape, "shape mismatch"

            # import ipdb
            # ipdb.set_trace()
            optimizer.zero_grad()
            loss = mse_loss(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader)
        training_time = time.time() - start_time
        print(
            f'Epoch: {epoch}, Average Loss: {epoch_loss}, training time : {training_time}'
        )

        # print("Epoch spend time ")
        total_loss = 0.0
        save_path = f"{args.save_path_root}/epoch{epoch}_{epoch_loss}.pth"
        torch.save(net, save_path)


def test(dataloader, args):
    for batch_idx, (data) in enumerate(dataloader):
        print(f'Batch Index: {batch_idx}')
        print(f'Data Shape: {data.shape}')

    pass


if __name__ == '__main__':

    main()
