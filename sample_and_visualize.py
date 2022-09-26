from torch import empty
from torch.nn import init
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from numpy import asarray
from matplotlib.pyplot import subplots


def sample_and_visualize(generator, latent_vec_size, img_ratio, num_imgs, device, trunc_theshold=0.1):
    noise = init.trunc_normal_(
            empty(num_imgs, latent_vec_size, img_ratio, 1, device=device),
            a=-trunc_theshold, b=trunc_theshold)
    fake = generator(noise)
    grid = make_grid(fake[:num_imgs].to(device), padding=2, normalize=True).cpu()
    show(grid)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
