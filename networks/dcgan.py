#!/usr/bin/env python
# coding: utf-8

# Note: This notebook was initially written following a tutorial from pytorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
import json
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
    def __init__(self, num_noise_vec_channels, base_num_out_channels, img_size, num_img_channels):
        super(Generator, self).__init__()

        factor = img_size // 8

        modules = [*self._FromLatent(num_noise_vec_channels, base_num_out_channels * factor)]

        while factor > 1:
            modules.extend(self._Block(base_num_out_channels * factor, base_num_out_channels * (factor // 2)))
            factor //= 2
        modules.extend(self._ToImage(base_num_out_channels, num_img_channels))

        self.main = nn.Sequential(*modules)

    def _FromLatent(self, num_lat_chan, num_out_chan):
        return nn.ConvTranspose2d(num_lat_chan, num_out_chan, kernel_size=4, stride=1, bias=False), \
               nn.BatchNorm2d(num_out_chan), \
               nn.ReLU(True)

    def _Block(self, num_in_chan, num_out_chan):
        return nn.ConvTranspose2d(num_in_chan, num_out_chan, kernel_size=4, stride=2, padding=1, bias=False), \
               nn.BatchNorm2d(num_out_chan), \
               nn.ReLU(True)

    def _ToImage(self, num_in_chan, num_img_chan):
        return nn.ConvTranspose2d(num_in_chan, num_img_chan, kernel_size=4, stride=2, padding=(1, 1), bias=False), \
               nn.Tanh()

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, img_size, img_ratio, num_img_channels, base_num_out_channels, padding_mode):
        super(Discriminator, self).__init__()

        max_factor = img_size // 8

        modules = [*self._FromImage(num_img_channels, base_num_out_channels, padding_mode, leakiness=0.2)]

        factor = 1
        while factor < max_factor:
            modules.extend(self._Block(base_num_out_channels * factor, base_num_out_channels * factor * 2,
                                       padding_mode, leakiness=0.2))
            factor *= 2
        modules.extend(self._Output(base_num_out_channels * factor, img_ratio))

        self.main = nn.Sequential(*modules)

    def _FromImage(self, num_img_chan, num_out_chan, padding_mode, leakiness):
        return nn.Conv2d(num_img_chan, num_out_chan, kernel_size=4, stride=2,
                         padding=1, padding_mode=padding_mode, bias=False), \
               nn.LeakyReLU(leakiness, inplace=True)

    def _Block(self, num_in_chan, num_out_chan, padding_mode, leakiness):
        return nn.Conv2d(num_in_chan, num_out_chan, kernel_size=4,
                         stride=2, padding=1, padding_mode=padding_mode, bias=False), \
               nn.BatchNorm2d(num_out_chan), \
               nn.LeakyReLU(leakiness, inplace=True),

    def _Output(self, num_in_chan, img_ratio):
        if img_ratio > 1:
            return nn.Conv2d(num_in_chan, 1, kernel_size=4, stride=1, bias=False), \
                   nn.Flatten(), \
                   nn.Linear(in_features=1 * img_ratio * 1, out_features=1), \
                   nn.Sigmoid()
        else:
            return nn.Conv2d(num_in_chan, 1, kernel_size=4, stride=1, bias=False), \
                   nn.Sigmoid()

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
              model_snap, model_to_load=None, show_graphs=True):

        out_dir = path.join(self.out_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir)
        self.last_out_dir = out_dir
        start_epoch = 0
        parameters = {"out_dir": self.out_dir,
                      "last_out_dir": self.last_out_dir,
                      "colormode": self.colormode,
                      "num_noise_vec_channels": self.num_noise_vec_channels,
                      "image_size_ratio": self.image_size_ratio,
                      "num_epochs": num_epochs,
                      "fake_img_snap": fake_img_snap,
                      "model_snap": model_snap,
                      "model_to_load": model_to_load}

        with open(out_dir + '/parameters.json', 'w') as file:
            json.dump(parameters, file)

        if model_to_load is not None:
            checkpoint = torch.load(model_to_load)
            generator.load_state_dict(checkpoint['generator_model_state_dict'])
            generator.train()
            discriminator.load_state_dict(checkpoint['discriminator_model_state_dict'])
            discriminator.train()
            self.optimizer_g.load_state_dict(checkpoint['generator_optimizer_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

        generator_losses = []
        discriminator_losses = []

        for epoch in range(start_epoch, (start_epoch + num_epochs)):
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
                    epoch, start_epoch + num_epochs - 1, d_error.item(), g_error.item(), D_x, D_G_z1, D_G_z2))
                generator_losses.append(g_error.item())
                discriminator_losses.append(d_error.item())

            if epoch % fake_img_snap == 0:
                utils.generate_and_save_images(out_dir, generator, epoch, self.noise_samples, self.colormode,
                                               show_graphs)
            if epoch % model_snap == 0:
                utils.save_checkpoint(out_dir, generator, self.optimizer_g, discriminator, self.optimizer_d, epoch)

            # Create loss graph
            utils.plot_loss_graph(discriminator_losses, generator_losses, out_dir, show_graphs)

        # Create gif
        utils.create_gif(out_dir)


# ---------------
#  Initialization
# ---------------

def create_gan(img_size, img_ratio, num_img_channels, num_noise_vec_channels, base_num_out_channels_g,
               base_num_out_channels_d, padding_mode, device):
    generator = Generator(num_noise_vec_channels, base_num_out_channels_g, img_size, num_img_channels).to(device)
    discriminator = Discriminator(img_size, img_ratio, num_img_channels,
                                  base_num_out_channels_d, padding_mode).to(device)
    generator.apply(utils.init_weights)
    discriminator.apply(utils.init_weights)

    return generator, discriminator


def create_trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, device):
    return Trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, device)
