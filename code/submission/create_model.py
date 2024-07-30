from pathlib import Path
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self,
                 depth=17,
                 n_channels=64,
                 image_channels=1,
                 use_bnorm=True):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(image_channels,
                      n_channels,
                      kernel_size,
                      padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(n_channels,
                          n_channels,
                          kernel_size,
                          padding=padding,
                          bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(n_channels,
                      image_channels,
                      kernel_size,
                      padding=padding,
                      bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = x - self.dncnn(x)
        return y


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


# def create_model(model_path: Path):
#     """Create and save an example DnCNN model"""
#     model = DnCNN()
#     example_input = torch.rand(1, 1, 540, 540)
#     jit_model = torch.jit.trace(model, example_inputs=example_input)
#     print(f'Saving model to: {model_path.absolute()}')
#     torch.jit.save(jit_model, model_path)


def create_model(model_path: Path):
    """Create and save an example jit model"""
    model = MapNN_No_Tconv()
    # model = torch.load("/Users/shantong/github/grandChallenge/AI4Life/code/AI4Life-MDC24-submission/resources/n2f_n2v_epoch18_0.07435575334562196.pth",map_location='cpu')
    model = torch.load("resources/s1_epoch73_tvloss.pth", map_location='cpu')
    example_input = torch.rand(1, 1, 255, 255)
    jit_model = torch.jit.trace(model, example_inputs=example_input)
    print(f'Saving model to: {model_path.absolute()}')
    torch.jit.save(jit_model, model_path)


if __name__ == "__main__":
    model_path = Path(__file__).parent / "resources/model.pth"
    create_model(model_path)
