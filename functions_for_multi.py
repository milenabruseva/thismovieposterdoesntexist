from pathlib import Path, PurePath
from PIL import Image


def transform_and_save_image(image_src_path, transform, output_folder_path):
    image = Image.open(image_src_path)
    image = transform(image)
    image.save(Path(output_folder_path) / PurePath(image_src_path).name)


def test_multi(image_src_path):
    return Path(image_src_path).mkdir(parents=True, exist_ok=True)
