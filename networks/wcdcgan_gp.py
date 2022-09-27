#!/usr/bin/env python
# coding: utf-8

# Note: This notebook was initially written following a tutorial from pytorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
import json
import os
from datetime import datetime
from os import path

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

import networks.utils as utils


# ----------
#  Networks
# ----------

class Generator(nn.Module):
    def __init__(self, num_noise_vec_channels, num_feature_vec_channels, base_num_out_channels, num_img_channels):
        super(Generator, self).__init__()
        self.deconv1_image = nn.ConvTranspose2d(num_noise_vec_channels, base_num_out_channels * 4, kernel_size=(4, 4),
                                                stride=(1, 1), bias=False)
        self.deconv1_image_bn = nn.BatchNorm2d(base_num_out_channels * 4)
        self.deconv1_label = nn.ConvTranspose2d(num_feature_vec_channels, base_num_out_channels * 4, kernel_size=(4, 4),
                                                stride=(1, 1), bias=False)
        self.deconv1_label_bn = nn.BatchNorm2d(base_num_out_channels * 4)

        self.deconv2 = nn.ConvTranspose2d(base_num_out_channels * 8, base_num_out_channels * 4, kernel_size=(4, 4),
                                          stride=(2, 2), padding=(1, 1), bias=False)
        self.deconv2_bn = nn.BatchNorm2d(base_num_out_channels * 4)

        self.deconv3 = nn.ConvTranspose2d(base_num_out_channels * 4, base_num_out_channels * 2, kernel_size=(4, 4),
                                          stride=(2, 2), padding=(1, 1), bias=False)
        self.deconv3_bn = nn.BatchNorm2d(base_num_out_channels * 2)

        self.deconv4 = nn.ConvTranspose2d(base_num_out_channels * 2, base_num_out_channels, kernel_size=(4, 4),
                                          stride=(2, 2), padding=(1, 1), bias=False)
        self.deconv4_bn = nn.BatchNorm2d(base_num_out_channels)

        self.deconv5 = nn.ConvTranspose2d(base_num_out_channels, num_img_channels, kernel_size=(4, 4), stride=(2, 2),
                                          padding=(1, 1), bias=False)

    def forward(self, input, label):
        x = F.relu(self.deconv1_image_bn(self.deconv1_image(input)))
        y = F.relu(self.deconv1_label_bn(self.deconv1_label(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        return torch.tanh(self.deconv5(x))


class Discriminator(nn.Module):
    def __init__(self, num_img_channels, num_feature_vec_channels, base_num_out_channels, padding_mode,
                 norm_layer_type):
        super(Discriminator, self).__init__()
        self.norm_layer_type = norm_layer_type
        if self.norm_layer_type == "none":
            self.conv1_1 = nn.Conv2d(num_img_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2),
                                     padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv1_2 = nn.Conv2d(num_feature_vec_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2),
                                     padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv2 = nn.Conv2d(base_num_out_channels * 2, base_num_out_channels * 4, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv3 = nn.Conv2d(base_num_out_channels * 4, base_num_out_channels * 8, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv4 = nn.Conv2d(base_num_out_channels * 8, base_num_out_channels * 16, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv5 = nn.Conv2d(base_num_out_channels * 16, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
            self.conv6 = nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)

        elif self.norm_layer_type == "layer":
            self.conv1_1 = nn.Conv2d(num_img_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2),
                                     padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv1_2 = nn.Conv2d(num_feature_vec_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2),
                                     padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv2 = nn.Conv2d(base_num_out_channels * 2, base_num_out_channels * 4, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv2_ln = nn.LayerNorm([base_num_out_channels * 4, 24, 16])

            self.conv3 = nn.Conv2d(base_num_out_channels * 4, base_num_out_channels * 8, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv3_ln = nn.LayerNorm([base_num_out_channels * 8, 12, 8])

            self.conv4 = nn.Conv2d(base_num_out_channels * 8, base_num_out_channels * 16, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv4_ln = nn.LayerNorm([base_num_out_channels * 16, 6, 4])

            self.conv5 = nn.Conv2d(base_num_out_channels * 16, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
            self.conv6 = nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)

        elif self.norm_layer_type == "instance":
            self.conv1_1 = nn.Conv2d(num_img_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2),
                                     padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv1_2 = nn.Conv2d(num_feature_vec_channels, base_num_out_channels, kernel_size=(4, 4), stride=(2, 2),
                                     padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv2 = nn.Conv2d(base_num_out_channels * 2, base_num_out_channels * 4, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv2_in = nn.InstanceNorm2d(base_num_out_channels * 4, affine=True)

            self.conv3 = nn.Conv2d(base_num_out_channels * 4, base_num_out_channels * 8, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv3_in = nn.InstanceNorm2d(base_num_out_channels * 8, affine=True)

            self.conv4 = nn.Conv2d(base_num_out_channels * 8, base_num_out_channels * 16, kernel_size=(4, 4),
                                   stride=(2, 2), padding=(1, 1), bias=False, padding_mode=padding_mode)
            self.conv4_in = nn.InstanceNorm2d(base_num_out_channels * 16, affine=True)

            self.conv5 = nn.Conv2d(base_num_out_channels * 16, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
            self.conv6 = nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1), bias=False)
        else:
            print("This layer norm is not implemented.")

    def forward(self, input, label):
        x = None
        if self.norm_layer_type == "none":
            x = F.leaky_relu(self.conv1_1(input), 0.2)
            y = F.leaky_relu(self.conv1_2(label), 0.2)
            x = torch.cat([x, y], 1)
            x = F.leaky_relu(self.conv2(x), 0.2)
            x = F.leaky_relu(self.conv3(x), 0.2)
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
        elif self.norm_layer_type == "layer":
            x = F.leaky_relu(self.conv1_1(input), 0.2)
            y = F.leaky_relu(self.conv1_2(label), 0.2)
            x = torch.cat([x, y], 1)
            x = F.leaky_relu(self.conv2_ln(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_ln(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.conv4_ln(self.conv4(x)))
            x = F.leaky_relu(self.conv5(x))
        elif self.norm_layer_type == "instance":
            x = F.leaky_relu(self.conv1_1(input), 0.2)
            y = F.leaky_relu(self.conv1_2(label), 0.2)
            x = torch.cat([x, y], 1)
            x = F.leaky_relu(self.conv2_in(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_in(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.conv4_in(self.conv4(x)))
            x = F.leaky_relu(self.conv5(x))
        else:
            print("This layer norm is not implemented.")
        return x


# ---------------
#  Training
# ---------------

class Trainer:
    def __init__(self, out_dir, num_samples, colormode,
                 num_noise_vec_channels, image_size_ratio, d_params, g_params,
                 learning_rate, beta1, beta2, n_critic, lambda_gp, device):
        self.out_dir = out_dir
        self.last_out_dir = None
        self.colormode = colormode
        self.num_noise_vec_channels = num_noise_vec_channels
        self.image_size_ratio = image_size_ratio
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.loss_function = nn.BCELoss()
        self.noise_samples = torch.randn(num_samples, num_noise_vec_channels, image_size_ratio, 1, device=device)
        self.optimizer_d = optim.Adam(d_params, lr=learning_rate, betas=(beta1, beta2))
        self.optimizer_g = optim.Adam(g_params, lr=learning_rate, betas=(beta1, beta2))

        self.real_label = 1.
        self.fake_label = 0.

    def train(self, generator, discriminator, dataloader, num_epochs, device, num_features, sample_labels_generator,
              fake_img_snap, model_snap, model_to_load, show_graphs=True):

        out_dir = path.join(self.out_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_dir)
        self.last_out_dir = out_dir

        start_epoch = 0
        parameters = {"out_dir": self.out_dir,
                      "last_out_dir": self.last_out_dir,
                      "colormode": self.colormode,
                      "num_noise_vec_channels": self.num_noise_vec_channels,
                      "image_size_ratio": self.image_size_ratio,
                      "n_critic": self.n_critic,
                      "lambda_gp": self.lambda_gp,
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
                noise = torch.randn(reals.size(0), self.num_noise_vec_channels,
                                    self.image_size_ratio, 1, device=device)

                input_features = torch.stack(data[6:43], dim=1).type(torch.FloatTensor).to(device)
                input_features_generator = input_features[:, :, None, None].expand(reals.size(0), num_features, 3, 1)
                input_features_discriminator = input_features[:, :, None, None].expand(reals.size(0), num_features, 96,
                                                                                       64)

                fakes = generator(noise, input_features_generator)
                d_real = discriminator(reals, input_features_discriminator).view(-1)
                d_fake = discriminator(fakes.detach(), input_features_discriminator).view(-1)

                grad_penalty = gradient_penalty(discriminator, reals, fakes, input_features_discriminator,
                                                device=device)

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
                    fakes = generator(noise, input_features_generator)
                    g_validity = discriminator(fakes, input_features_discriminator).view(-1)

                    g_loss = -g_validity.mean()
                    g_loss.backward()
                    self.optimizer_g.step()

                    D_G_z2 = g_validity.mean().item()

                pbar.set_description('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (
                    epoch, start_epoch + num_epochs - 1, d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
                generator_losses.append(g_loss.item())
                discriminator_losses.append(d_loss.item())

            if epoch % fake_img_snap == 0:
                utils.generate_and_save_images(out_dir, generator, epoch, self.noise_samples, self.colormode,
                                               show_graphs, sample_labels_generator)
            if epoch % model_snap == 0:
                utils.save_checkpoint(out_dir, generator, self.optimizer_g, discriminator, self.optimizer_d, epoch)

            # Create loss graph
            utils.plot_loss_graph(discriminator_losses, generator_losses, out_dir, show_graphs)
        # Create gif
        utils.create_gif(out_dir)


def gradient_penalty(discriminator, reals, fakes, input_features_discriminator, device):
    BATCH_SIZE, C, H, W = reals.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), device=device).repeat(1, C, H, W)
    interpolated_imgs = (reals * alpha + fakes * (1 - alpha)).requires_grad_(True)

    d_out = discriminator(interpolated_imgs, input_features_discriminator)

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

def create_gan(num_img_channels, num_noise_vec_channels, num_feature_vec_channels, base_num_out_channels_g,
               base_num_out_channels_d,
               d_norm_layer_type, padding_mode, device):
    generator = Generator(num_noise_vec_channels, num_feature_vec_channels, base_num_out_channels_g,
                          num_img_channels).to(device)
    discriminator = Discriminator(num_img_channels, num_feature_vec_channels, base_num_out_channels_d, padding_mode,
                                  d_norm_layer_type).to(device)
    generator.apply(utils.init_weights)
    discriminator.apply(utils.init_weights)

    return generator, discriminator


def create_trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, n_critic, lambda_gp, device):
    return Trainer(out_dir, num_samples, colormode, num_noise_vec_channels, image_size_ratio,
                   d_params, g_params, learning_rate, beta1, beta2, n_critic, lambda_gp, device)
