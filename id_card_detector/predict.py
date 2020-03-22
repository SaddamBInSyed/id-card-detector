import torch
import numpy as np
from albumentations import (
    Compose,
    Normalize
)
from id_card_detector.download_model import download_model

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_resnet50_model(num_classes: int):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_maskrcnn_resnet50_model_with_weights():
    """
    Loads model and its weights, then returns it.
    """
    model_name = "maskrcnn_resnet50"
    # downlaod model if not present
    WEIGHT_PATH = download_model(model_name)
    # load model dict
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(f=WEIGHT_PATH, map_location=DEVICE)
    # load cfg from model dict
    cfg = model_dict["cfg"]
    # load model
    model = get_maskrcnn_resnet50_model(num_classes=cfg["NUM_CLASSES"])
    # load weights
    model.load_state_dict(model_dict["state_dict"])
    # return loaed model
    return model


def get_transform() -> Compose:
    transforms = Compose([Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    return transforms


def to_float_tensor(img: np.array) -> torch.tensor:
    # Converts numpy images to pytorch format
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def get_maskrcnn_prediction(image: np.array, model,
                            category_mapping: dict = {},
                            threshold: float = 0.5,
                            verbose: int = 1) -> (list, list, list):
    # apply transform
    transforms = get_transform()
    augmented = transforms(image=image)
    image = augmented["image"]
    # convert to tensor
    image = to_float_tensor(image).unsqueeze(0)

    # get prediction
    model.eval()
    pred = model(image)

    # map prediction ids to labels if category_mapping is given as input
    if not(category_mapping == {}):
        INSTANCE_CATEGORY_NAMES = category_mapping
    else:
        INSTANCE_CATEGORY_NAMES = {ind: ind for ind in range(999)}

    # get predictions with above threshold prediction scores
    pred_score = list(pred[0]['scores'].detach().numpy())
    num_predictions_above_threshold = sum([1 for x in pred_score if x > threshold])
    pred_num = num_predictions_above_threshold

    pred_masks, pred_boxes, pred_classes, pred_scores = [], [], [], []
    # process predictions if there are any
    if pred_num > 0:
        # get mask predictions
        pred_masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        if len(pred_masks.shape) == 3:
            pred_masks = pred_masks[:pred_num]
        elif len(pred_masks.shape) == 2:
            pred_masks = np.expand_dims(pred_masks, 0)

        # get box predictions
        pred_boxes = [[[int(i[0]), int(i[1])], [int(i[2]), int(i[3])]] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_boxes = pred_boxes[:pred_num]

        # get class predictions
        pred_classes = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_classes = pred_classes[:pred_num]

        # get prediction scores
        pred_scores = list(pred[0]['scores'].detach().numpy())

    # print the number of detections
    if verbose == 1:
        print("There are {} detected instances.".format(pred_num))

    return pred_masks, pred_boxes, pred_classes, pred_scores


def get_prediction(image: np.array,
                   model_name: str = "maskrcnn_resnet50",
                   threshold: float = 0.75,
                   verbose: int = 1) -> (list, list, list):

    if model_name == "maskrcnn_resnet50":
        # load selected model
        model = get_maskrcnn_resnet50_model_with_weights()
        category_mapping = {1: "id_card"}

        # perform prediction
        (masks,
         bboxes,
         classes,
         scores) = get_maskrcnn_prediction(image,
                                           model,
                                           category_mapping,
                                           threshold,
                                           verbose)
    else:
        TypeError("Model: {} not supported".format(model_name))
        model = None

    return masks, bboxes, classes, scores

