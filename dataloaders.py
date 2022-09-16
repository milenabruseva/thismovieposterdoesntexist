import os

import torch
import torchvision
from torchvision import transforms

import pandas as pd

from PIL import Image

img_transform_train = transforms.Compose([...])

class PosterDataset(torch.utils.data.Dataset):
    def __int__(self, table_path, img_root_path, img_transform = None):
        self.table = pd.read_pickle(table_path)
        self.img_root_path = img_root_path
        self.img_transform = img_transform

        # Prepare Table
        self.table.reset_index(level=0, inplace=True)
        self.table['image_path'] = '.'+ os.sep + 'data' + os.sep + self.table.id.astype('string') + '.jpg'

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx: int):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        _, _, _, og_lang, pop, year, runtime, vote_count, _, _, is_thriller, is_horror, is_animation, is_scifi, is_action,\
        is_drama, is_fantasy, is_adventure, is_family, is_comedy, is_tv, is_crime, is_mystery, is_war, is_romance,\
        is_music, is_history, is_docu, is_western, _ = self.table.iloc[[idx]].values.flatten()
        year = year.year

        image = pil_loader(self.table.image_path[idx])
        if self.img_transform is not None:
            image = self.img_transform(image)

        return image, og_lang, pop, year, runtime, vote_count, is_thriller, is_horror, is_animation, is_scifi, is_action,\
               is_drama, is_fantasy, is_adventure, is_family, is_comedy, is_tv, is_crime, is_mystery, is_war,\
               is_romance, is_music, is_history, is_docu, is_western




def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGB")
