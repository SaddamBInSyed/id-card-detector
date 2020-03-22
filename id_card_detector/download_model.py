import os
import argparse
from pathlib import Path
from id_card_detector.utils import download

# get maskrcnn-resnet50 path
home_path = str(Path.home())
weight_path = os.path.join(home_path,
                           "id_card_detector",
                           "weights",
                           "maskrcnn-resnet50.pt")

WEIGHT_DICT = {"maskrcnn_resnet50": {"url": "https://github.com/fcakyon/id-card-detector/releases/download/v0.0.1/maskrcnn-resnet50.pt",
                                     "downlaod_dir": weight_path}
               }


def download_model(model_name: str):
    """
    This script downloads desired model weights from github repo.
    """
    # get required dirs/paths
    download_url = WEIGHT_DICT[model_name]["url"]
    file_name = download_url.split("/")[-1]
    download_dir = WEIGHT_DICT[model_name]["downlaod_dir"]
    file_path = download_dir + file_name

    # check if model is already present and download model file if not present
    if os.path.isfile(weight_path) is not True:
        print("Craft text detector weight will be downloaded to {}".format(weight_path))
        download(url=download_url, save_dir=download_dir)

    return file_path


if __name__ == '__main__':
    # construct the argument parser
    ap = argparse.ArgumentParser()

    # add the arguments to the parser
    ap.add_argument("model_name", default="maskrcnn_resnet50", help="Model name to be downloaded.")
    args = vars(ap.parse_args())

    # download model
    _ = download_model(args['model_name'])
