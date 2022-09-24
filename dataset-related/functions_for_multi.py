from pathlib import Path, PurePath
from PIL import Image


def transform_and_save_image(image_src_path: str, transform, output_folder_path: str, imagetype="jpg"):
    image = Image.open(image_src_path)
    image = image.convert("RGB")
    image = transform(image)
    if imagetype == "jpg":
        image.save(Path(output_folder_path) / PurePath(image_src_path).name)
    elif imagetype == "png":
        image.save(Path(output_folder_path) / PurePath(image_src_path.removesuffix(".jpg") + ".png").name, format='png',
                   compress_level=0, optimize=False)


def test_multi(image_src_path):
    return Path(image_src_path).mkdir(parents=True, exist_ok=True)
