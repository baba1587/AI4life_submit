"""
Import
"""

import torch
import torch.nn as nn
import tifffile
import numpy as np
import argparse
import os, random, torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as ts
from tqdm import tqdm
import time
import torch.nn.functional as F

# enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# random seed definition
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def getTiffImage(image_path):
    with tifffile.TiffFile(image_path) as tif:
        images = tif.asarray()
        return images


def getImageInfo(image):
    print(image.shape)
    print(image.min())
    print(image.max())


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

    def addGaussianNoise(image, mean, stddev=0.1):
        gaussian_noise = torch.normal(mean, stddev, size=image.size())
        return torch.clamp(image + gaussian_noise, 0.0, 1.0)

    def genMask2D(sample):
        mask_ratio = args.mask_ratio
        assert 0 <= mask_ratio <= 1, "mask_ratio should in [0, 1] "
        C, H, W = sample.shape
        # calculate masked values num
        num_pixels = H * W
        num_mask_pixels = int(mask_ratio * num_pixels)

        mask = torch.zeros(H * W, dtype=torch.float32)
        mask_indices = torch.randperm(num_pixels)[:num_mask_pixels]

        mask[mask_indices] = 1.0
        mask = mask.view(H, W)
        mask = mask.unsqueeze(0).expand(C, -1, -1)

        # generate mask image
        masked_image = sample * (1 - mask)

        return masked_image, mask

    def train_transforms(sample):
        transforms = ts.Compose([
            ts.RandomCrop(args.train_size),
            ts.RandomHorizontalFlip(p=0.5),
            ts.RandomVerticalFlip(p=0.5),
            ts.RandomRotation(degrees=90),
        ])

        transformed_label = transforms(sample)

        # add noise
        # gaussian_mean = 0.0
        # uniform_dist = torch.distributions.Uniform(0.0, 0.2)
        # gaussian_std = float(uniform_dist.sample((1, 1)))
        # transformed_input = addGaussianNoise(transformed_label, gaussian_mean,
        #                                      gaussian_std)

        # n2fTransform_sample = n2fTransform(
        # transformed_label.squeeze())  # return list of transformed image

        # for data in n2fTransform_sample:
        #     # data_info = n2fTransform_sample[index]
        #     input = data[0]
        #     label = data[1]
        #     masked_input, mask = genMask2D(input)
        #     data[0] = masked_input
        #     data.append(mask)

        # n2v method
        masked_input, mask = genMask2D(transformed_label)

        return [masked_input, transformed_label, mask]

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


"""
Model
"""


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


"""
Loss
"""


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h +
                                         w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def n2v_loss(
    manipulated_patches: torch.Tensor,
    original_patches: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    N2V Loss function described in A Krull et al 2018.

    Parameters
    ----------
    manipulated_patches : torch.Tensor
        Patches with manipulated pixels.
    original_patches : torch.Tensor
        Noisy patches.
    masks : torch.Tensor
        Array containing masked pixel locations.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    errors = (original_patches - manipulated_patches)**2
    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss


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
        print(
            f"dataset has {len(images)} slices with the original shape {images.shape}"
        )
        return images

    def __getitem__(self, index):
        train_image = self.images[index]
        train_image = AI4LifeTransform(train_image, args=self.args)

        return train_image


def main():
    args = parse_args()

    # set up
    set_seed()

    DATA_ROOT = args.data_root

    path_u1 = DATA_ROOT + "/u1.tiff"
    path_u2 = DATA_ROOT + "/u2.tiff"
    path_s1 = DATA_ROOT + "/s1.tiff"
    path_s2 = DATA_ROOT + "/s2.tiff"

    dataset_u1 = AI4LifeDataset(dataset_path=path_u1, args=args)
    # dataset_u2 = AI4LifeDataset(dataset_path=path_u2, args=args)
    # dataset = torch.utils.data.ConcatDataset([dataset_u1, dataset_u2])
    dataloader = DataLoader(dataset=dataset_u1,
                            batch_size=args.train_batchsize,
                            shuffle=True,
                            num_workers=args.numworkers,
                            pin_memory=True,
                            drop_last=False)

    train(dataloader=dataloader, args=args)
    print("training complete!")


def train(dataloader, args):
    total_loss = 0.0
    # bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    net = MapNN_No_Tconv(out_ch=32).to(args.device)
    tvloss = TVLoss()
    print("model load done ...")
    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.2)

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

        for batch_idx, (data) in tqdm(enumerate(dataloader)):
            # index = np.random.randint(0, len(data))
            # data_index = data[index]
            train_input = data[0].to(args.device)
            label = data[1].to(args.device)
            mask = data[2].to(args.device)
            # print(f'Batch Index: {batch_idx}')
            # print(f'Data Shape: {input.shape} and label shape : {label.shape}')

            output = net(train_input)
            # assert output.shape == input.shape, "shape mismatch"

            # import ipdb
            # ipdb.set_trace()
            optimizer.zero_grad()
            ratio = 0.7
            tv_ratio = 0.1
            loss = ratio * n2v_loss(
                output, label, mask) + (1 - ratio - tv_ratio) * l1_loss(
                    output * mask, label * mask) + tv_ratio * tvloss(output)
            # loss = ratio * n2v_loss(output, label, mask) + (
            # 1 - ratio) * l1_loss(output * mask, label * mask)# + tv_ratio * tvloss(output)
            # loss = n2v_loss(output, label, mask)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        learning_schedular.step()
        epoch_loss = total_loss / len(dataloader)
        training_time = time.time() - start_time
        print(
            f'Epoch: {epoch}, Average Loss: {epoch_loss}, training time : {training_time}, lr_rate : {learning_schedular.get_last_lr()[0]}'
        )

        total_loss = 0.0
        save_path = f"{args.save_path_root}/epoch{epoch}_{epoch_loss}.pth"
        torch.save(net, save_path)


def test(dataloader, args):
    for batch_idx, (data) in enumerate(dataloader):
        print(f'Batch Index: {batch_idx}')
        print(f'Data Shape: {data.shape}')

    pass


################################
# args
################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_epoch", type=int, default=200)
    parser.add_argument("--train_batchsize", type=int, default=128)
    parser.add_argument("--numworkers", type=int, default=12)
    """mask ratio for n2v"""
    parser.add_argument("--mask_ratio", type=float, default=0.4)

    parser.add_argument("--data_root", type=str, default=" ")
    parser.add_argument("--save_path_root", type=str, default=" ")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    main()
