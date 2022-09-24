import glob
import numpy as np
from PIL import Image

import torch
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def sample_posters(dataloader, number_of_samples, colormode, device):
    sample_reals = next(iter(dataloader))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")

    if colormode == "RGB":
        plt.imshow(np.transpose(
            vutils.make_grid(sample_reals[0].to(device)[:number_of_samples], padding=2, normalize=True).cpu(),
            (1, 2, 0)))
    elif colormode == "HSV":
        plt.imshow(hsv_to_rgb(np.transpose(
            vutils.make_grid(sample_reals[0].to(device)[:number_of_samples], padding=2, normalize=True).cpu(),
            (1, 2, 0))))
    else:
        print('Colormode not implemented for image visualization.')

    return sample_reals


def generate_and_save_images(output_dir, generator, epoch, noise, colormode, show=True):
    with torch.no_grad():
        images = generator(noise).detach().cpu()
    img_list = []

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    img_list.append(vutils.make_grid(images, padding=2, normalize=True))

    if colormode == "RGB":
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    elif colormode == "HSV":
        plt.imshow(hsv_to_rgb(np.transpose(img_list[-1], (1, 2, 0))))
    else:
        print('Colormode not implemented for image visualization.')

    fig.savefig(output_dir + '/image_at_epoch_{:04d}.png'.format(epoch))

    if show:
        plt.show();
    else:
        plt.close();


def create_gif(directory):
    sample_images = [Image.open(image) for image in glob.glob(f"{directory}/image*.png")]
    thumbnail = sample_images[0]
    thumbnail.save(f"{directory}/animation.gif", format="GIF", append_images=sample_images, save_all=True,
                   duration=1000,
                   loop=0)


def save_checkpoint(output_dir, generator_network, optimizer_generator, discriminator_network, optimizer_discriminator,
                    epoch):
    torch.save({
        'epoch': epoch,
        'generator_model_state_dict': generator_network.state_dict(),
        'generator_optimizer_state_dict': optimizer_generator.state_dict(),
        'discriminator_model_state_dict': discriminator_network.state_dict(),
        'discriminator_optimizer_state_dict': optimizer_discriminator.state_dict(),
    }, output_dir + '/gan_at_epoch_{:04d}.pt'.format(epoch))
