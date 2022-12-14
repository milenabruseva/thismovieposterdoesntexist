import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

img_transform_train = transforms.Compose([...])


class PosterDataset(torch.utils.data.Dataset):
    def __init__(self, table_path, img_root_path, img_transform=None,
                 colormode='RGB', img_in_ram=False,
                 genre=None, genre_logic='and', og_lang=None, year=None, runtime=None,
                 max_num=None, sort=None):
        '''

        :param table_path: Path to the table pickle file.
        :param img_root_path: Path to the image folder.
        :param img_transform: Transformation for the images.
        :param colormode: pillow colormode, e.g. "RGB", "LAB", "HSV"
        :param img_in_ram: Whether to already load images into RAM.
        :param genre: A single genre or list of genres.
        :param genre_logic: 'and' or 'or'.
        :param og_lang: A single original language.
        :param year: A single year, 2-tuple for span or list of spans.
        :param runtime: A 2-tuple for min to max runtime. (np.inf for not upper bound)
        :param max_num: Maximal number of posters to retrieve.
        :param sort: If number of posters exceed maximal number of posters, sorting method for cutoff: 'popularity', 'vote_count'.
        '''

        self.table = pd.read_pickle(table_path)
        self.img_root_path = img_root_path
        self.img_transform = img_transform
        self.colormode = colormode
        self.img_in_ram = img_in_ram
        self.images = []

        ### Prepare Table
        # Genre
        if genre is not None:
            if isinstance(genre, str):
                self.table = self.table[self.table[genre]]
            elif isinstance(genre, (tuple, list)) and all(isinstance(x, str) for x in genre):
                if genre_logic == 'and':
                    for gen in genre:
                        self.table = self.table[self.table[gen]]
                elif genre_logic == 'or':
                    tot_gen = self.table[genre[0]]
                    for gen in genre[1:]:
                        tot_gen = tot_gen | self.table[gen]
                else:
                    print('Genre: genre_logic not implemented.')
            else:
                print('Genre: Unknown kind of type.')

        # Original Language
        if og_lang is not None:
            self.table = self.table[self.table.original_language == og_lang]

        # Year Span
        self.table['year'] = pd.DatetimeIndex(self.table.release_date).year
        self.table.drop(columns=['release_date'])
        if year is not None:
            if isinstance(year, int):
                self.table = self.table[self.table.year == year]
            elif isinstance(year, (tuple, list)):
                if all(isinstance(x, int) for x in year):
                    self.table = self.table[self.table.year.between(*year)]  # year is tuple or list of ints
                elif all(isinstance(x, (tuple, list)) and all(isinstance(y, int) for y in x) for x in year):
                    in_year_spans = (self.table.year < 0)
                    for span in multispan:
                        in_year_spans = in_year_spans | self.table.year.between(*span)
                    self.table = self.table[in_year_spans]  # year is tuple or list of tuple or list of ints
                else:
                    print('Year: Unknown kind of tuple.')
            else:
                print('Year: Unknown kind of type.')

        # Runtime
        if runtime is not None:
            self.table = self.table[self.table.runtime.between(*runtime)]

        # Cutoff
        if max_num is not None:
            if sort is not None:
                self.table.sort_values(by=sort, ascending=False, inplace=True)
            self.table = self.table.iloc[:max_num]

        ### Reset index to 0,1,2,...
        self.table.reset_index(level=0, inplace=True)

        ### Image Path
        self.table['image_path'] = self.img_root_path + os.sep + self.table.id.astype('string') + '.jpg'

        ### Load images into RAM; todo: optimize loading and tranforming
        if self.img_in_ram:
            for path in self.table.image_path:
                if self.img_transform is not None:
                    self.images.append(self.img_transform(pil_loader(path, self.colormode)))
                else:
                    self.images.append(pil_loader(path))

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx: int):
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()

        _, _, _, og_lang, pop, _, runtime, vote_count, _, _, is_thriller, is_horror, is_animation, is_scifi, is_action, \
        is_drama, is_fantasy, is_adventure, is_family, is_comedy, is_tv, is_crime, is_mystery, is_war, is_romance, \
        is_music, is_history, is_docu, is_western, lang_ar, lang_cn, lang_de, lang_el, lang_en, lang_es, lang_fr, \
        lang_hi, lang_it, lang_ja, lant_ko, lang_other, lang_pt, lang_ru, lang_sv, lang_tl, lang_tr, \
        lang_zh, year, _ = self.table.iloc[[idx]].values.flatten()

        if self.img_in_ram:
            image = self.images[idx]
        else:
            image = pil_loader(self.table.image_path[idx], self.colormode)
            if self.img_transform is not None:
                image = self.img_transform(image)

        return image, og_lang, pop, year, runtime, vote_count, is_thriller, is_horror, is_animation, is_scifi, is_action, \
               is_drama, is_fantasy, is_adventure, is_family, is_comedy, is_tv, is_crime, is_mystery, is_war, \
               is_romance, is_music, is_history, is_docu, is_western, lang_ar, lang_cn, lang_de, lang_el, lang_en, \
               lang_es, lang_fr, lang_hi, lang_it, lang_ja, lant_ko, lang_other, lang_pt, lang_ru, lang_sv, \
               lang_tl, lang_tr, lang_zh


def pil_loader(path: str, colormode="RGB") -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(colormode)
