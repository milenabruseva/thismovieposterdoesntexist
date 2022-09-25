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
            nn.ConvTranspose2d(num_noise_vec_channels, base_num_out_channels * 8, kernel_size=(4, 4), stride=(1, 1),
                               bias=False),
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
            nn.ConvTranspose2d(base_num_out_channels * 2, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(base_num_out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_num_out_channels, num_img_channels, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_img_channels, base_num_out_channels, padding_mode, norm_layer_type):
        super(Discriminator, self).__init__()
        if norm_layer_type == "none":
            self.main = nn.Sequential(
                nn.Conv2d(num_img_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                          padding_mode=padding_mode, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels, base_num_out_channels * 2, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 2, base_num_out_channels * 4, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 4, base_num_out_channels * 8, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 8, 1, kernel_size=(4, 4), stride=(1, 1), bias=False),
                nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
            )
        elif norm_layer_type == "layer":
            self.main = nn.Sequential(
                nn.Conv2d(num_img_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                          padding_mode=padding_mode, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels, base_num_out_channels * 2, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.LayerNorm([base_num_out_channels * 2, 24, 16]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 2, base_num_out_channels * 4, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.LayerNorm([base_num_out_channels * 4, 12, 8]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 4, base_num_out_channels * 8, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.LayerNorm([base_num_out_channels * 8, 6, 4]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 8, 1, kernel_size=(4, 4), stride=(1, 1), bias=False),
                nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
            )
        elif norm_layer_type == "instance":
            self.main = nn.Sequential(
                nn.Conv2d(num_img_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                          padding_mode=padding_mode, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels, base_num_out_channels * 2, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.InstanceNorm2d(base_num_out_channels * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 2, base_num_out_channels * 4, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.InstanceNorm2d(base_num_out_channels * 4, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 4, base_num_out_channels * 8, kernel_size=(4, 4), stride=(2, 2),
                          padding=(1, 1), padding_mode=padding_mode, bias=False),
                nn.InstanceNorm2d(base_num_out_channels * 8, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_num_out_channels * 8, 1, kernel_size=(4, 4), stride=(1, 1), bias=False),
                nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
            )
        else:
            print("This layer norm is not implemented.")

    def forward(self, input):
        return self.main(input)


# ---------------
#  Training
# ---------------

class Trainer:
    def __init__(self, out_dir, num_samples, colormode,
                 num_noise_vec_channels, image_size_ratio, d_params, g_params,
                 learning_rate, beta1, beta2,
                 n_critic, lambda_gp,
                 device):
        self.out_dir = out_dir
        self.last_out_dir = None
        self.colormode = colormode
        self.num_noise_vec_channels = num_noise_vec_channels
        self.image_size_ratio = image_size_ratio
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.noise_samples = torch.randn(num_samples, num_noise_vec_channels, image_size_ratio, 1, device=device)
        self.optimizer_d = optim.Adam(d_params, lr=learning_rate, betas=(beta1, beta2))
        self.optimizer_g = optim.Adam(g_params, lr=learning_rate, betas=(beta1, beta2))

    def train(self, generator, discriminator, dataloader, num_epochs, device, fake_img_snap,
              model_snap, show_graphs=True):

        out_dir = path.join(self.out_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir)
        self.last_out_dir = out_dir

        generator_losses = []
        discriminator_losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(pbar := tqdm(dataloader)):
                # Sample images
                reals = data[0].to(device)
                noise = torch.randn(reals.size(0), self.num_noise_vec_channels,
                                    self.image_size_ratio, 1, device=device)
                fakes = generator(noise)

                # Train Discriminator
                discriminator.zero_grad()

                d_real = discriminator(reals).view(-1)
                d_fake = discriminator(fakes).view(-1)

                grad_penalty = gradient_penalty(discriminator, reals, fakes, device=device)

                d_loss = d_fake.mean() - d_real.mean() + self.lambda_gp * grad_penalty
                d_loss.backward()
                self.optimizer_d.step()

                D_x = d_real.mean().item()
                D_G_z1 = d_fake.mean().item()

                # Train Generator every n_critic-th iteration
                if i % self.n_critic == 0:
                    generator.zero_grad()

                    noise = torch.randn(reals.size(0), self.num_noise_vec_channels,
                                        self.image_size_ratio, 1, device=device)
                    fakes = generator(noise)
                    g_validity = discriminator(fakes).view(-1)

                    g_loss = -g_validity.mean()
                    g_loss.backward()
                    self.optimizer_g.step()

                    D_G_z2 = g_validity.mean().item()

                pbar.set_description('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                    epoch, num_epochs - 1, d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
                generator_losses.append(g_loss.item())
                discriminator_losses.append(d_loss.item())

            if epoch % fake_img_snap == 0:
                utils.generate_and_save_images(out_dir, generator, epoch, self.noise_samples, self.colormode,
                                               show_graphs)
            if epoch % model_snap == 0:
                utils.save_checkpoint(out_dir, generator, self.optimizer_g, discriminator, self.optimizer_d, epoch)

        # Create loss graph
        utils.plot_loss_graph(discriminator_losses, generator_losses, out_dir, show_graphs)

        # Create gif
        utils.create_gif(out_dir)


def gradient_penalty(discriminator, reals, fakes, device):
    BATCH_SIZE, C, H, W = reals.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), device=device).repeat(1, C, H, W)
    interpolated_imgs = (reals * alpha + fakes * (1 - alpha)).requires_grad_(True)

    d_out = discriminator(interpolated_imgs)

    gradient = torch.autograd.grad(
        inputs=interpolated_imgs,
        outputs=d_out,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    return ((gradient.norm(2, dim=1) - 1) ** 2).mean()


# ---------------
#  Initialization
# ---------------

def create_gan(num_img_channels, num_noise_vec_channels, base_num_out_channels_g, base_num_out_channels_d,
               d_norm_layer_type, padding_mode, device):
    generator = Generator(num_noise_vec_channels, base_num_out_channels_g, num_img_channels).to(device)
    discriminator = Discriminator(num_img_channels, base_num_out_channels_d, padding_mode, d_norm_layer_type).to(device)
    generator.apply(utils.init_weights)
    discriminator.apply(utils.init_weights)

    return generator, discriminator


def create_trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, n_critic, lambda_gp, device):
    return Trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, n_critic, lambda_gp, device)
