#!/usr/bin/env python
# coding: utf-8

# Note: This notebook was initially written following a tutorial from pytorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
import os
from os import path
from datetime import datetime
import json

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
#from tqdm.auto import tqdm

import networks.utils as utils


# ----------
#  Networks
# ----------

class Generator(nn.Module):
    def __init__(self, num_noise_vec_channels, num_feature_vec_channels, base_num_out_channels,
                 img_size, num_img_channels):
        super(Generator, self).__init__()

        factor = img_size // 8

        # Image input
        self.in_img = nn.Sequential(*self._FromLatent(num_noise_vec_channels,
                                                      base_num_out_channels * (factor // 2)))
        # Feature input
        self.in_lab = nn.Sequential(*self._FromLatent(num_feature_vec_channels,
                                                      base_num_out_channels * (factor // 2)))

        modules = []
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

    def forward(self, image, label):
        return self.main(torch.cat([self.in_img(image), self.in_lab(label)], 1))


class Discriminator(nn.Module):
    def __init__(self, img_size, img_ratio, num_img_channels, num_feature_vec_channels,
                 base_num_out_channels, padding_mode):
        super(Discriminator, self).__init__()

        max_factor = img_size // 4

        # Image input
        self.in_img = nn.Sequential(*self._FromImage(num_img_channels, base_num_out_channels,
                                                     padding_mode, leakiness=0.2))
        # Feature input
        self.in_lab = nn.Sequential(*self._FromImage(num_feature_vec_channels, base_num_out_channels,
                                                     padding_mode, leakiness=0.2))

        modules = []
        factor = 2
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
                   nn.Linear(in_features=1 * img_ratio * 1, out_features=1)
        else:
            return nn.Conv2d(num_in_chan, 1, kernel_size=4, stride=1, bias=False)

    def forward(self, image, label):
        return self.main(torch.cat([self.in_img(image), self.in_lab(label)], 1))


# ---------------
#  Training
# ---------------

class Trainer:
    def __init__(self, out_dir, num_samples, colormode,
                 num_noise_vec_channels, image_size, image_size_ratio, d_params, g_params,
                 learning_rate, beta1, beta2, device):
        self.out_dir = out_dir
        self.last_out_dir = None
        self.colormode = colormode
        self.num_noise_vec_channels = num_noise_vec_channels
        self.image_size = image_size
        self.image_size_ratio = image_size_ratio

        self.noise_samples = nn.init.trunc_normal_(
            torch.empty(num_samples, num_noise_vec_channels, image_size_ratio, 1, device=device),
            a=-0.1, b=0.1)
        self.optimizer_d = optim.Adam(d_params, lr=learning_rate, betas=(beta1, beta2))
        self.optimizer_g = optim.Adam(g_params, lr=learning_rate, betas=(beta1, beta2))

    def train(self, generator, discriminator, dataloader, num_epochs, device,
              num_features, sample_labels_generator,
              fake_img_snap, model_snap, model_to_load=None, show_graphs=True):

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

        start_with_d = True

        for epoch in range(start_epoch, (start_epoch + num_epochs)):
            for i, data in enumerate(dataloader):
                # Generate inputs
                reals = data[0].to(device)
                noise = torch.randn(reals.size(0), self.num_noise_vec_channels,
                                    self.image_size_ratio, 1, device=device)
                # Labels
                input_features = torch.stack(data[6:(6 + num_features)], dim=1).type(torch.FloatTensor).to(device)
                input_features_generator = input_features[:, :, None, None].expand(
                    reals.size(0), num_features, self.image_size_ratio, 1)
                input_features_discriminator = input_features[:, :, None, None].expand(
                    reals.size(0), num_features, self.image_size_ratio * self.image_size // 2, self.image_size)
                fakes = generator(noise, input_features_generator)

                if start_with_d:
                    # Train Discriminator
                    discriminator.zero_grad()

                    d_reals = discriminator(reals, input_features_discriminator).view(-1)
                    d_fakes = discriminator(fakes.detach(), input_features_discriminator).view(-1)

                    loss_d = (((d_reals - d_fakes.mean() - 1) ** 2).mean() +
                              ((d_fakes - d_reals.mean() + 1) ** 2).mean()) / 2

                    loss_d.backward()
                    self.optimizer_d.step()

                    #D_x = d_reals.mean().item()
                    #D_G_z1 = d_fakes.mean().item()

                else:

                    # Train Generator
                    generator.zero_grad()

                    d_reals = discriminator(reals, input_features_discriminator).view(-1)
                    d_fakes = discriminator(fakes, input_features_discriminator).view(-1)

                    loss_g = (((d_reals - d_fakes.mean() + 1) ** 2).mean() +
                              ((d_fakes - d_reals.mean() - 1) ** 2).mean())/2

                    loss_g.backward()
                    self.optimizer_g.step()

                    #D_G_z2 = d_fakes.mean().item()

                if i % 2 == 0 and i > 0:
                    #pbar.set_description('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                    #    epoch, start_epoch + num_epochs - 1, loss_d.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))
                    generator_losses.append(loss_g.item())
                    discriminator_losses.append(loss_d.item())

                start_with_d = not start_with_d

            print("Epoch "+str(epoch)+"finished.")

            if epoch % fake_img_snap == 0:
                utils.generate_and_save_images(out_dir, generator, epoch, self.noise_samples, self.colormode,
                                               show_graphs, sample_labels_generator)
            if epoch % model_snap == 0:
                utils.save_checkpoint(out_dir, generator, self.optimizer_g, discriminator, self.optimizer_d, epoch)

        # Create loss graph
        utils.plot_loss_graph(discriminator_losses, generator_losses, out_dir, show_graphs)

        # Create gif
        utils.create_gif(out_dir)


# ---------------
#  Initialization
# ---------------

def create_gan(img_size, img_ratio, num_img_channels, num_noise_vec_channels, num_feature_vec_channels,
               base_num_out_channels_g, base_num_out_channels_d, padding_mode, device):
    generator = Generator(num_noise_vec_channels, num_feature_vec_channels, base_num_out_channels_g,
                          img_size, num_img_channels).to(device)
    discriminator = Discriminator(img_size, img_ratio, num_img_channels, num_feature_vec_channels,
                                  base_num_out_channels_d, padding_mode).to(device)
    generator.apply(utils.init_weights)
    discriminator.apply(utils.init_weights)

    return generator, discriminator


def create_trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size,
                   image_size_ratio, d_params, g_params, learning_rate, beta1, beta2, device):
    return Trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size,
                   image_size_ratio, d_params, g_params, learning_rate, beta1, beta2, device)
