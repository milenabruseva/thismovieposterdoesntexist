import glob
import random
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from matplotlib.colors import hsv_to_rgb
from torch import nn


def sample_posters(dataloader, number_of_samples, colormode, device):
    """
    The sample_posters function takes a dataloader and number of samples as input. It then plots the first
    number_of_samples images from the dataloader in a grid. The function returns sample_reals, which is an array
    containing a sample of real images from the dataloader.

    :param dataloader: Retrieve the training data
    :param number_of_samples: Specify the number of samples to be displayed
    :param colormode: Select the color space in which the images are shown
    :param device: Indicate whether the model is trained on a gpu or cpu
    :return: A batch of data from the dataloader
    """
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


def generate_and_save_images(output_dir, generator, epoch, noise, colormode, show=True, sample_labels_generator=None):
    """
    The generate_and_save_images function generates and saves a grid of images.

    :param output_dir: Specify the location where the generated images are saved
    :param generator: Generator to use
    :param epoch: Specify the number of epochs that have been performed
    :param noise: Noise to be passed to the generator
    :param colormode: Determine which colormode to use for the image visualization
    :param show=True: Plot the image
    :param sample_labels_generator=None: Use if model is conditional
    """
    with torch.no_grad():
        if sample_labels_generator is None:
            images = generator(noise).detach().cpu()
        else:
            images = generator(noise, sample_labels_generator).detach().cpu()
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
    """
    The create_gif function creates a gif from the images in the specified directory.
    The function takes one argument, which is the name of a directory containing all of
    the images to be used in creating the GIF. The function returns nothing.

    :param directory: Specify the directory where the images are stored
    """
    sample_images = [Image.open(image) for image in glob.glob(f"{directory}/image*.png")]
    thumbnail = sample_images[0]
    thumbnail.save(f"{directory}/animation.gif", format="GIF", append_images=sample_images, save_all=True,
                   duration=1000,
                   loop=0)


def save_checkpoint(output_dir, generator_network, optimizer_generator, discriminator_network, optimizer_discriminator,
                    epoch):
    """
    The save_checkpoint function saves the model's state dictionary, optimizer state dictionary, and current epoch number
    in a checkpoint file.

    :param output_dir: Specify the path to the directory where we want to save our model
    :param generator_network: Generator model to save
    :param optimizer_generator: Generator optimizer to save
    :param discriminator_network: Discriminator model to save
    :param optimizer_discriminator: Discriminator optimizer to save
    :param epoch: Save the checkpoint at a specific epoch, used for naming purposes
    """
    torch.save({
        'epoch': epoch,
        'generator_model_state_dict': generator_network.state_dict(),
        'generator_optimizer_state_dict': optimizer_generator.state_dict(),
        'discriminator_model_state_dict': discriminator_network.state_dict(),
        'discriminator_optimizer_state_dict': optimizer_discriminator.state_dict(),
    }, output_dir + '/gan_at_epoch_{:04d}.pt'.format(epoch))


def plot_loss_graph(discriminator_losses, generator_losses, out_dir, show_graphs):
    """
    The plot_loss_graph function plots the loss of both the discriminator and generator.
    It saves this graph to a file named 'training_loss.png' in the output directory specified by out_dir.

    :param discriminator_losses: Plot the discriminator loss
    :param generator_losses: Plot the generator loss
    :param out_dir: Specify the directory to save the generated loss graph
    :param show_graphs: Show the graph
    """
    fig = plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss at End of Training")
    plt.plot(generator_losses, label="Generator")
    plt.plot(discriminator_losses, label="Discriminator")
    plt.xlabel("Total Batch Iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig(path.join(out_dir, "training_loss.png"))
    if show_graphs:
        plt.show();
    else:
        plt.close();


def plot_grid(generator, noise_vector_size, device, seed):
    """
    The plot_grid function takes a generator, noise vector size, device and seed as input.
    It then creates a grid of images using the generator.
    The number of images in each row is equal to the number of languages present in our dataset
    and the number of rows - to the number of genres.

    :param generator: Generator to use
    :param noise_vector_size: Size of the noise vector
    :param device: Specify whether you want to run on a cpu or gpu
    :param seed: Set the random seed for reproducibility
    """
    number_of_samples = 1
    genres = {"is_thriller": 0, "is_horror": 0, "is_animation": 0, "is_scifi": 0,
              "is_action": 0, "is_drama": 0, "is_fantasy": 0, "is_adventure": 0,
              "is_family": 0, "is_comedy": 0, "is_tv": 0, "is_crime": 0,
              "is_mystery": 0, "is_war": 0, "is_romance": 0, "is_music": 0,
              "is_history": 0, "is_docu": 0, "is_western": 0}
    languages = {"lang_ar": 0, "lang_cn": 0, "lang_de": 0, "lang_el": 0,
                 "lang_en": 0, "lang_es": 0, "lang_fr": 0, "lang_hi": 0,
                 "lang_it": 0, "lang_ja": 0, "lant_ko": 0, "lang_other": 0,
                 "lang_pt": 0, "lang_ru": 0, "lang_sv": 0, "lang_tl": 0,
                 "lang_tr": 0, "lang_zh": 0}

    samples = []
    for genre in genres:
        current_genres = genres.copy()
        current_genres[genre] = 1
        for language in languages:
            current_languages = languages.copy()
            current_languages[language] = 1
            features = current_genres | current_languages
            feature_vectors = [torch.ones(number_of_samples) * is_on for is_on in features.values()]
            feature_vectors = torch.stack(feature_vectors, 1)
            samples.append(
                feature_vectors[:, :, None, None].expand(number_of_samples, len(features), 3, 1).to(device))

    samples = torch.reshape(torch.stack(samples, dim=1),
                            (len(genres) * len(languages), len(genres) + len(languages), 3, 1))
    random.seed(seed)
    torch.manual_seed(seed)
    noise = torch.randn(1, noise_vector_size, 3, 1, device=device)
    noise = noise.expand(len(genres) * len(languages), noise_vector_size, 3, 1).to(device)
    with torch.no_grad():
        images = generator(noise, samples).detach().cpu()
    img = vutils.make_grid(images, nrow=len(languages), padding=2, normalize=True)
    fig = plt.figure(figsize=(20, 30))
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    fig.savefig("" + "image.png")


def init_weights(m):
    """
    The init_weights function initializes the weights of a network.
    It is called before training begins.

    :param m: The model parameters
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
