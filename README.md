
[![PyPI version](https://badge.fury.io/py/craft-text-detector.svg)](https://badge.fury.io/py/craft-text-detector)
![CI](https://github.com/fcakyon/craft-text-detector/workflows/CI/badge.svg)

## ID Card Detector
Everything you need for ID card detection. Easily perform ID card detection/segmentation, fit quad to detected mask and unwarp the ID card surface to rectangle with Pytorch models on Linux&Windows. Works on all types of ID cards.
 
 **Package maintainer: Fatih Cagatay Akyon**
 
### Overview
Perform instance segmentation for ID cards with pretrained MaskRCNN model and get predicted bounding boxes, masks and prediction scores. Works on CPU/GPU and Linux/Windows.

<img width="1000" alt="teaser" src="./figures/idcarddetector_example.gif">


## Getting started
### Installation
```console
pip install id-card-detector
```

### Basic Usage
```python
# import package
import id_card_detector

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# perform id card detection and export results to output directory
_,_,_,_ = id_card_detector.detect_card(image_path=image_path,
				       output_dir=output_dir
				       unwarp=True)
```

### Advanced Usage
```python
# import package
import id_card_detector as card_det

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# read image
image = card_det.read_image(image_path)

# get prediction
masks, boxes, classes, scores = card_det.get_prediction(image=image, threshold=0.75)

# export a visual of detected bboxes and masks
prediction_visual = visualize_prediction(image, masks, boxes, classes,
                                         rect_th=2,
                                         text_size=0.85,
                                         text_th=2,
                                         color=color,
                                         output_dir=output_dir)

# export detected bounding boxes
export_predicted_bboxes(image=image,
                        boxes=boxes,
                        output_dir=output_dir)

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
```

