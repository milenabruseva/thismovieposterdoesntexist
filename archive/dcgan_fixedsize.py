#!/usr/bin/env python
# coding: utf-8

# Note: This notebook was initially written following a tutorial from pytorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

import os
from datetime import datetime
from os import path

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

import networks.utils as utils


# ----------
#  Networks
# ----------

class Generator(nn.Module):
    def __init__(self, num_noise_vec_channels, base_num_out_channels, num_img_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(num_noise_vec_channels, base_num_out_channels * 8, kernel_size=(4, 4),
                               stride=(1, 1), bias=False),
            nn.BatchNorm2d(base_num_out_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_num_out_channels * 8, base_num_out_channels * 4, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(base_num_out_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_num_out_channels * 4, base_num_out_channels * 2, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(base_num_out_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_num_out_channels * 2, base_num_out_channels, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(base_num_out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_num_out_channels, num_img_channels, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_img_channels, base_num_out_channels, padding_mode):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_img_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      padding_mode=padding_mode, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_num_out_channels, base_num_out_channels * 2, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(base_num_out_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_num_out_channels * 2, base_num_out_channels * 4, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(base_num_out_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_num_out_channels * 4, base_num_out_channels * 8, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(base_num_out_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_num_out_channels * 8, 1, kernel_size=(4, 4), stride=(1, 1), bias=False),
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# ---------------
#  Training
# ---------------

class Trainer:
    def __init__(self, out_dir, num_samples, colormode,
                 num_noise_vec_channels, image_size_ratio, d_params, g_params,
                 learning_rate, beta1, beta2, device):
        self.out_dir = out_dir
        self.last_out_dir = None
        self.colormode = colormode
        self.num_noise_vec_channels = num_noise_vec_channels
        self.image_size_ratio = image_size_ratio

        self.loss_function = nn.BCELoss()
        self.noise_samples = torch.randn(num_samples, num_noise_vec_channels, image_size_ratio, 1, device=device)
        self.optimizer_d = optim.Adam(d_params, lr=learning_rate, betas=(beta1, beta2))
        self.optimizer_g = optim.Adam(g_params, lr=learning_rate, betas=(beta1, beta2))

        self.real_label = 1.
        self.fake_label = 0.

    def train(self, generator, discriminator, dataloader, num_epochs, device, fake_img_snap,
              model_snap, show_graphs=True):

        out_dir = path.join(self.out_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir)
        self.last_out_dir = out_dir

        generator_losses = []
        discriminator_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(pbar := tqdm(dataloader)):
                # Train Discriminator
                # on reals
                discriminator.zero_grad()
                reals = data[0].to(device)
                labels = torch.full((reals.size(0),), self.real_label, dtype=torch.float, device=device)
                output_d = discriminator(reals).view(-1)
                d_error_on_reals = self.loss_function(output_d, labels)
                d_error_on_reals.backward()
                D_x = output_d.mean().item()

                # on fakes
                noise = torch.randn(reals.size(0), self.num_noise_vec_channels,
                                    self.image_size_ratio, 1, device=device)
                fakes = generator(noise)
                labels.fill_(self.fake_label)
                output_d = discriminator(fakes.detach()).view(-1)
                d_error_on_fakes = self.loss_function(output_d, labels)
                d_error_on_fakes.backward()
                D_G_z1 = output_d.mean().item()
                d_error = d_error_on_reals + d_error_on_fakes
                self.optimizer_d.step()

                # Train Generator
                generator.zero_grad()
                labels.fill_(self.real_label)
                g_output = discriminator(fakes).view(-1)
                g_error = self.loss_function(g_output, labels)
                g_error.backward()
                D_G_z2 = g_output.mean().item()
                self.optimizer_g.step()

                pbar.set_description('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                    epoch, num_epochs - 1, d_error.item(), g_error.item(), D_x, D_G_z1, D_G_z2))
                generator_losses.append(g_error.item())
                discriminator_losses.append(d_error.item())

            if epoch % fake_img_snap == 0:
                utils.generate_and_save_images(out_dir, generator, epoch, self.noise_samples, self.colormode, show_graphs)
            if epoch % model_snap == 0:
                utils.save_checkpoint(out_dir, generator, self.optimizer_g, discriminator, self.optimizer_d, epoch)

        # Create loss graph
        utils.plot_loss_graph(discriminator_losses, generator_losses, out_dir, show_graphs)

        # Create gif
        utils.create_gif(out_dir)


# ---------------
#  Initialization
# ---------------

def create_gan(num_img_channels, num_noise_vec_channels, base_num_out_channels_g, base_num_out_channels_d,
               padding_mode, device):
    generator = Generator(num_noise_vec_channels, base_num_out_channels_g, num_img_channels).to(device)
    discriminator = Discriminator(num_img_channels, base_num_out_channels_d, padding_mode).to(device)
    generator.apply(utils.init_weights)
    discriminator.apply(utils.init_weights)

    return generator, discriminator


def create_trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, device):
    return Trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, device)
