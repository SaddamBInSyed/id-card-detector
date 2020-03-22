import os
import cv2
import copy
import random
import numpy as np

from id_card_detector.utils import create_dir


def select_random_color():
    """
    Selects random color.
    """
    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255],
              [255, 255, 0], [255, 0, 255], [80, 70, 180], [250, 80, 190],
              [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    return colors[random.randrange(0, 10)]


def apply_color_mask(image: np.array, color: tuple):
    """
    Applies color mask to given input image.
    """
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    (r[image == 1],
     g[image == 1],
     b[image == 1]) = color
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def visualize_prediction(image: str,
                         masks, boxes, classes,
                         rect_th: float = 3,
                         text_size: float = 3,
                         text_th: float = 3,
                         color: tuple = (0, 0, 0),
                         output_dir: str = "output/",
                         file_name: str = "inference_result"):
    """
    Visualizes prediction classes, bounding boxes, masks over the source image
    and exports it to output folder.
    """
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # create output folder if not present
    create_dir(output_dir)
    # select random color if not specified
    if color == (0, 0, 0):
        color = select_random_color()
    # add bbox and mask to image if present
    if len(masks) > 0:
        for i in range(len(masks)):
            rgb_mask = apply_color_mask(masks[i], color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.7, 0)
            cv2.rectangle(image, tuple(boxes[i][0]), tuple(boxes[i][1]),
                          color=color, thickness=rect_th)
            # arange bounding box text location
            if boxes[i][0][1] - 10 > 10:
                boxes[i][0][1] -= 10
            else:
                boxes[i][0][1] += 10
            # add bounding box text
            cv2.putText(image, classes[i], tuple(boxes[i][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, color,
                        thickness=text_th)
    # save inference result
    save_path = os.path.join(output_dir, file_name + ".png")
    cv2.imwrite(save_path,
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def crop_inference_bbox(image: np.array,
                        boxes: list,
                        output_dir: str = "output/",
                        file_name: str = "cropped_inference_result"):
    """
    Crops the predicted bounding box regions and exports them to output folder.
    """
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # create output folder if not present
    create_dir(output_dir)
    # crop detections
    if len(boxes) > 0:
        for ind in range(len(boxes)):
            cropped_img = image[int(boxes[ind][0][1]):int(boxes[ind][1][1]),
                                int(boxes[ind][0][0]):int(boxes[ind][1][0]),
                                :]
            save_path = os.path.join(output_dir,
                                     file_name + "_" + str(ind) + ".png")
            cv2.imwrite(save_path,
                        cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def warp_quads_to_rects(image, quads,
                        output_dir: str = "output/",
                        file_name: str = "warped"):
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # create output folder if not present
    create_dir(output_dir)
    # warp quads to rects
    warped_rects = []
    for ind, quad in enumerate(quads):
        pts = np.array(quad, dtype="float32")
        # warp four point region to rectangle
        warped_rect = four_point_transform(image, pts)
        warped_rects.append(warped_rect)

        # export the warped region
        save_path = os.path.join(output_dir,
                                 file_name + "_" + str(ind) + ".png")
        cv2.imwrite(save_path, cv2.cvtColor(warped_rect, cv2.COLOR_RGB2BGR))
    # return the warped images
    return warped_rects


def fit_quads_to_masks(image,
                       masks,
                       verbose: bool = True,
                       color: tuple = (0, 0, 0),
                       output_dir: str = "output/",
                       file_name: str = "quad_fit"):
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # create output folder if not present
    create_dir(output_dir)
    # select random color if not specified
    if color == (0, 0, 0):
        color = select_random_color()
    # fit quads to masks
    quads = []
    for ind, mask in enumerate(masks):
        mask = mask.astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(mask,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        epsilon = 0.01*cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        quad = [[point_[0], point_[1]] for point in approx for point_ in point]
        quads.append(quad)

        cv2.drawContours(image, [approx], 0, color, 3)
        if verbose:
            print("Simplified contours for mask-{} has {} points."
                  .format(ind, len(approx)))

    # export the image with drawns quads on top of it
    save_path = os.path.join(output_dir, file_name + ".png")
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # return the fitted quads
    return quads
