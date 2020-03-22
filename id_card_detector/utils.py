import os
import json
import zipfile
import jsonschema
from tqdm import tqdm
from urllib.request import urlretrieve

image_schema = {
    "type": "object",
    "properties": {
        "file_name": {
            "type": "string"
            },
        "id": {
            "type": "integer"
            }
    },
    "required": ["file_name", "id"]
}

segmentation_schema = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {
            "type": "number",
            },
        "additionalItems": False
        },
    "additionalItems": False
}

annotation_schema = {
    "type": "object",
    "properties": {
        "image_id": {
            "type": "integer"
            },
        "category_id": {
            "type": "integer"
            },
        "segmentation": segmentation_schema
    },
    "required": ["image_id", "category_id", "segmentation"]
}

category_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
            },
        "id": {
            "type": "integer"
            }
    },
    "required": ["name", "id"]
}

coco_schema = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": image_schema,
            "additionalItems": False
            },
        "annotations": {
            "type": "array",
            "items": annotation_schema,
            "additionalItems": False
            },
        "categories": {
            "type": "array",
            "items": category_schema,
            "additionalItems": False
            }
    },
    "required": ["images", "annotations", "categories"]
}


def read_and_validate_coco_annotation(
        coco_annotation_path: str) -> (dict, bool):
    """
    Reads coco formatted annotation file and validates its fields.
    """
    try:
        with open(coco_annotation_path) as json_file:
            coco_dict = json.load(json_file)
        jsonschema.validate(coco_dict, coco_schema)
        response = True
    except jsonschema.exceptions.ValidationError as e:
        print("well-formed but invalid JSON:", e)
        response = False
    except json.decoder.JSONDecodeError as e:
        print("poorly-formed text, not JSON:", e)
        response = False

    return coco_dict, response


def process_coco(coco_file_path: str) -> (list, dict):
    """
    Accepts a coco object detection file.
    Returns list of images and categories.
    """
    # check if coco file is valid and read it
    (coco_dict,
     response) = read_and_validate_coco_annotation(coco_file_path)

    # raise error if coco file is not valid
    if not(response):
        raise TypeError

    # rearrange coco file for better annotation reach
    images = list()
    for image in coco_dict["images"]:
        image_annotations = list()
        for annotation in coco_dict["annotations"]:
            if image["id"] == annotation["image_id"]:
                image_annotations.append(annotation)
        image["annotations"] = image_annotations
        images.append(image)

    return images, coco_dict["categories"]


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def get_category_mapping_froom_coco_file(coco_file_path: str) -> dict:
    """
    Creates category id>name mapping from a coco annotation file.
    """
    # check if coco file is valid and read it
    (coco_dict,
     response) = read_and_validate_coco_annotation(coco_file_path)

    # raise error if coco file is not valid
    if not(response):
        raise TypeError

    coco_categories = coco_dict["categories"]
    category_mapping = {
            coco_category["id"]: coco_category["name"] for
            coco_category in coco_categories
            }
    return category_mapping


class TqdmUpTo(tqdm):
    """
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    https://pypi.org/project/tqdm/#hooks-and-callbacks
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download(url: str, save_dir: str):
    """
    Downloads file by http request, shows remaining time.
    https://pypi.org/project/tqdm/#hooks-and-callbacks
    Example inputs:
        url: 'ftp://smartengines.com/midv-500/dataset/01_alb_id.zip'
        save_dir: 'data/'
    """

    # create save_dir if not present
    create_dir(save_dir)
    # download file
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                  desc=url.split('/')[-1]) as t:  # all optional kwargs
        urlretrieve(url, filename=os.path.join(save_dir, url.split('/')[-1]),
                    reporthook=t.update_to, data=None)


def unzip(file_path: str, dest_dir: str):
    """
    Unzips compressed .zip file.
    Example inputs:
        file_path: 'data/01_alb_id.zip'
        dest_dir: 'data/'
    """

    # unzip file
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(dest_dir)
