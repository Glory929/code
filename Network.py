
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from scipy.ndimage import gaussian_filter

from blocks import *


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):

        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1 / 3)

        self.sigma_kernels = [
            self.get_gaussian_kernel_3d(2 * i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel_3d(self, kernel_size, sigma):
        kernel_weights = np.zeros((kernel_size, kernel_size, kernel_size))
        kernel_weights[kernel_size // 2, kernel_size // 2, kernel_size // 2] = 1

        kernel_weights = gaussian_filter(kernel_weights, sigma=sigma, mode='constant')

        kernel_weights /= np.sum(kernel_weights)  # 归一化
        # 核大小应该是 c,1,h,w,d
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().to(device)

    def forward(self, x):
        G = x

        # Level 1
        L0 = rearrange(G, 'b c h w d -> b c (h w d)')
        L0_att = F.softmax(L0, dim=2) @ L0.transpose(1, 2)
        L0_att = F.softmax(L0_att, dim=-1)

        # Next Levels
        attention_maps = [L0_att]
        pyramid = [G]

        for kernel in self.sigma_kernels:
            G = F.conv3d(input=G, weight=kernel, bias=None, padding='same', groups=self.in_channels)
            pyramid.append(G)

        for i in range(1, self.pyramid_levels):
            L = torch.sub(pyramid[i - 1], pyramid[i])
            L = rearrange(L, 'b c h w d -> b c (h w d)')
            L_att = F.softmax(L, dim=2) @ L.transpose(1, 2)
            attention_maps.append(L_att)

        return sum(attention_maps)


class GBANet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,
                 do_res: bool = False,
                 do_res_up_down: bool = False,
                 checkpoint_style: bool = None,
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],
                 norm_type='group',
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        self.stem = nn.Conv3d(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = GBANetDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.enc_block_1 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = GBANetDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.enc_block_2 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = GBANetDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.enc_block_3 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = GBANetDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.bottleneck = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = GBANetUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.dec_block_3 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = GBANetUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.dec_block_2 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = GBANetUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
        )

        self.dec_block_1 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = GBANetUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type
        )

        self.dec_block_0 = nn.Sequential(*[
            GBANetBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type
            )
            for i in range(block_counts[8])]
                                         )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes)
            self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes)

        self.block_counts = block_counts

        # --------跨层连接----------#
        self.myup2 = nn.Sequential(
            nn.Conv3d(n_channels * 8, n_channels * 4, kernel_size=1),  # 通道减半
            nn.Upsample(scale_factor=2, mode='nearest')  # 大小翻倍
        )
        self.myup1 = nn.Sequential(
            nn.Conv3d(n_channels * 8, n_channels * 2, kernel_size=1),  # 通道减半
            nn.Upsample(scale_factor=4, mode='nearest')  # 大小翻倍
        )
        self.myup0 = nn.Sequential(
            nn.Conv3d(n_channels * 8, n_channels, kernel_size=1),  # 通道减半
            nn.Upsample(scale_factor=8, mode='nearest')  # 大小翻倍
        )
        # --------LaplacianPyramid----------#
        self.freq0 = LaplacianPyramid(in_channels=n_channels * 1, pyramid_levels=3)
        self.freq1 = LaplacianPyramid(in_channels=n_channels * 2, pyramid_levels=3)
        self.freq2 = LaplacianPyramid(in_channels=n_channels * 4, pyramid_levels=3)
        self.freq3 = LaplacianPyramid(in_channels=n_channels * 8, pyramid_levels=3)

        self.q_conv0 = nn.Conv3d(in_channels=n_channels * 1, out_channels=n_channels * 1, kernel_size=1)
        self.q_conv1 = nn.Conv3d(in_channels=n_channels * 2, out_channels=n_channels * 2, kernel_size=1)
        self.q_conv2 = nn.Conv3d(in_channels=n_channels * 4, out_channels=n_channels * 4, kernel_size=1)
        self.q_conv3 = nn.Conv3d(in_channels=n_channels * 8, out_channels=n_channels * 8, kernel_size=1)

        # self.cat_conv0 = nn.Conv3d(in_channels=n_channels*2, out_channels=n_channels*1, kernel_size=1)
        # self.cat_conv1 = nn.Conv3d(in_channels=n_channels*4, out_channels=n_channels*2, kernel_size=1)
        # self.cat_conv2 = nn.Conv3d(in_channels=n_channels*8, out_channels=n_channels*4, kernel_size=1)
        # self.cat_conv3 = nn.Conv3d(in_channels=n_channels*16, out_channels=n_channels*8, kernel_size=1)


    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):

        x = self.stem(x)  # stem in_c=4 --> n_channel=32

        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3  # 删除（释放）变量

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            b, c, h, w, d = x.size()  # stem x
            x_res_0 = self.enc_block_0(x)
            queries = F.softmax(self.q_conv0(x).reshape(b, c, h * w * d), dim=1)
            freq_0 = self.freq0(x)  # b c c
            freq_attmap = (freq_0 @ queries).reshape(b, c, h, w, d)
            attention = x_res_0 + freq_attmap
            x = self.down_0(attention)
            # print(freq_0.shape, freq_attmap.shape, x.shape)
            # torch.Size([1, 32, 32]) torch.Size([1, 32, 160, 160, 128]) torch.Size([1, 64, 80, 80, 64])

            b, c, h, w, d = x.size()
            x_res_1 = self.enc_block_1(x)
            queries = F.softmax(self.q_conv1(x).reshape(b, c, h * w * d), dim=1)
            freq_1 = self.freq1(x)  # b c c
            freq_attmap = (freq_1 @ queries).reshape(b, c, h, w, d)
            attention = x_res_1 + freq_attmap
            x = self.down_1(attention)
            # print(freq_1.shape, freq_attmap.shape, x.shape)
            # torch.Size([1, 64, 64]) torch.Size([1, 64, 80, 80, 64]) torch.Size([1, 128, 40, 40, 32])

            b, c, h, w, d = x.size()
            x_res_2 = self.enc_block_2(x)
            queries = F.softmax(self.q_conv2(x).reshape(b, c, h * w * d), dim=1)
            freq_2 = self.freq2(x)  # b c c
            freq_attmap = (freq_2 @ queries).reshape(b, c, h, w, d)
            attention = x_res_2 + freq_attmap
            x = self.down_2(attention)
            # print(freq_2.shape, freq_attmap.shape, x.shape)
            # torch.Size([1, 128, 128]) torch.Size([1, 128, 40, 40, 32]) torch.Size([1, 256, 20, 20, 16])

            b, c, h, w, d = x.size()
            x_res_3 = self.enc_block_3(x)
            queries = F.softmax(self.q_conv3(x).reshape(b, c, h * w * d), dim=1)
            freq_3 = self.freq3(x)  # b c c
            freq_attmap = (freq_3 @ queries).reshape(b, c, h, w, d)
            attention = x_res_3 + freq_attmap
            x = self.down_3(attention)
            # print(freq_3.shape, freq_attmap.shape, x.shape)
            # torch.Size([1, 256, 256]) torch.Size([1, 256, 20, 20, 16]) torch.Size([1, 512, 10, 10, 8])

            x = self.bottleneck(x)

            if self.do_ds:  # 深监督
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            # ---------跨层连接--------#
            up2x = self.myup2(x_up_3)
            up1x = self.myup1(x_up_3)
            up0x = self.myup0(x_up_3)

            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)

            if self.do_ds:  # 深监督
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            dec_x = dec_x * up2x  #
            x = self.dec_block_2(dec_x)

            if self.do_ds:  # 深监督
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            dec_x = dec_x * up1x  #
            x = self.dec_block_1(dec_x)

            if self.do_ds:  # 深监督
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            dec_x = dec_x * up0x  #
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


if __name__ == "__main__":
    network = GBANet(
        in_channels=4,
        n_channels=32,
        n_classes=4,
        # exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        exp_r = 2,
        kernel_size=3,
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        block_counts = [2,2,2,2,2,2,2,2,2],
        checkpoint_style=None,

    ).cuda()


    with torch.no_grad():
        print(network)
        # x = torch.zeros((2, 4, 128, 128, 128)).cuda()
        # print(network(x)[0].shape)
        x = torch.zeros((1, 4, 160, 160, 128)).cuda()
        print('output shape:', network(x)[0].shape)
