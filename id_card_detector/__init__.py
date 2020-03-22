from __future__ import absolute_import

__version__ = "0.1.1"

from id_card_detector.cv_utils import (read_image,
                                       visualize_prediction,
                                       export_predicted_bboxes,
                                       fit_quads_over_masks,
                                       visualize_quads,
                                       unwarp_quads,
                                       export_unwarped_quads)

from id_card_detector.predict import (get_prediction)


def detect_card(image_path: str,
                output_dir: str = "output/",
                unwarp: bool = True,
                model_name: str = "maskrcnn_resnet50",
                color: tuple = (0, 0, 0)):
    """
    Arguments:
        image_path: path to the image to be processed
        output_dir: path to the results to be exported
        unwarp: unwarp detected id card to rectangle
        model_name: model to be used in the inference
        color: color to be used in the mask/bbox/quad visualizations
    """

    # read image from given path
    image = read_image(image_path)

    # get prediction
    masks, boxes, classes, scores = get_prediction(image=image,
                                                   model_name="maskrcnn_resnet50",
                                                   threshold=0.75)

    # visualize detected bboxes and masks
    prediction_visual = visualize_prediction(image, masks, boxes, classes,
                                             rect_th=2,
                                             text_size=0.85,
                                             text_th=2,
                                             color=color,
                                             output_dir=output_dir)

    if not unwarp:
        # export detected bounding boxes
        export_predicted_bboxes(image=image,
                                boxes=boxes,
                                output_dir=output_dir)

        # arange other values as empty
        quads = []
        unwarped_quads = []
    else:
        # fit quads to predicted masks
        quads = fit_quads_over_masks(image, masks)

        # visualize/export quads
        quad_visual = visualize_quads(image=image,
                                      quads=quads,
                                      output_dir=output_dir,
                                      color=color)

        # unwarp quads to rects
        unwarped_quads = unwarp_quads(image, quads)

        # export unwarped quads
        export_unwarped_quads(unwarped_quads,
                              output_dir=output_dir)

    return masks, boxes, classes, scores, quads
