import cv2
import os
import random
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def gaussian2D(shape, sigma=1):
    """
    2D Gaussian function
    Args:
        shape: (diameter, diameter)
        sigma: variance

    Returns: h

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

def draw_gaussian(heatmap, center, radius, k=1):
    """
    Get a heatmap of one class
    Args:
        heatmap: The heatmap of one class(storage in single channel)
        center: The location of object center
        radius: 2D Gaussian circle radius
        k: The magnification of the Gaussian

    Returns: heatmap

    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def gaussian_radius(det_size, min_overlap=0.3):
    """
    Get gaussian circle radius.
    Args:
        det_size: (height, width)
        min_overlap: overlap minimum

    Returns: radius

    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def drop_cap_bbox(bboxes, texts):
    return [bboxes[i] for i, text in enumerate(texts) if len(text) < 2]

def aug_retangle(image, bboxes, num_mask = 1):
    # drop_cap_box = [bboxes[i] for i, text in enumerate(texts) if len(text) < 2]
    if num_mask >= len(bboxes):
        num_mask = 1
    mask_bbox = random.sample(bboxes, num_mask)
    for mask in mask_bbox:
        x = int(mask[0])
        y = int(mask[1])
        w = int(mask[2])
        h = int(mask[3])
        pixel_value = image[y, x]
        cv2.rectangle(image, 
                    (x, y), 
                    (x+w, y+h), 
                    (int(pixel_value), int(pixel_value), int(pixel_value)), 
                    thickness=-1)
    
    drop_mask_bboxes = [item for item in bboxes if item not in mask_bbox]
    return image, drop_mask_bboxes

def aug_circle(image, bboxes, num_mask = 1):
    # drop_cap_box = [bboxes[i] for i, text in enumerate(texts) if len(text) < 2]
    if num_mask >= len(bboxes):
        num_mask = 1
    mask_bbox = random.sample(bboxes, num_mask)
    for mask in mask_bbox:
        x = int(mask[0])
        y = int(mask[1])
        w = int(mask[2])
        h = int(mask[3])
        min_crop_value = int(np.min(image[y:y+h, x:x+w]))
        radius = 5
        if (w < 7) or (h < 7) :
            radius = 3

        cv2.circle(image, ((x+x+w)//2, (y+y+h)//2), radius, 
                   (min_crop_value, min_crop_value, min_crop_value), -1)
    
    drop_mask_bboxes = [item for item in bboxes if item not in mask_bbox]
    return image, drop_mask_bboxes

class ImgAugTransform:
    def __init__(self):
        self.aug_shift= iaa.PadToFixedSize(width=100, height=100)
        self.aug_brightness = iaa.Add((-5, 15))
        self.aug_blur = iaa.GaussianBlur(sigma=(0.1, 1.0))
        self.aug_fliplr = iaa.Fliplr(1.0)
        self.aug_affline_rot = iaa.Affine(scale={"x": (0.5, 0.9), "y": (0.5, 0.9)})
        self.aug_crop = iaa.CropAndPad(percent=(-0.25, 0.25))

    def decode_from_iaa(self, iaa_bboxes):
        aug_bboxes = []
        for  i in range(len(iaa_bboxes.bounding_boxes)):
            aug_bboxes.append([iaa_bboxes[i].x1, iaa_bboxes[i].y1, iaa_bboxes[i].x2, iaa_bboxes[i].y2])
        return aug_bboxes
      
    def __call__(self, img, bboxes):
        '''
        img: cv2 mat
        bboxes: [[x1, y1, x2, y2], .....]
        '''
        boundingbox = []
        for bbox in bboxes:
            boundingbox.append(BoundingBox(x1=bbox[0],
                                           y1=bbox[1],
                                           x2=bbox[2],
                                           y2=bbox[3]))
        bbs = BoundingBoxesOnImage(boundingbox, shape=img.shape)
        aug = random.randint(0, 5)
        if aug == 0:
            img = img
            aug_bboxes = bboxes
        if aug == 1:
            img, bboxes = self.aug_brightness(image=img, bounding_boxes=bbs)
            aug_bboxes =  self.decode_from_iaa(bboxes)
        if aug == 2:
            img, bboxes = self.aug_blur(image=img, bounding_boxes=bbs)
            aug_bboxes =  self.decode_from_iaa(bboxes)
        if aug == 3:
            img, bboxes = self.aug_fliplr(image=img, bounding_boxes=bbs)
            aug_bboxes =  self.decode_from_iaa(bboxes)
        if aug == 4:
            img, bboxes = self.aug_affline_rot(image=img, bounding_boxes=bbs)
            aug_bboxes =  self.decode_from_iaa(bboxes)
        if aug == 5:
            img, bboxes = self.aug_shift(image=img, bounding_boxes=bbs)
            aug_bboxes =  self.decode_from_iaa(bboxes)

        return  img, aug_bboxes
    
class Mosaic:
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_image_and_bboxes(self, image, bboxes, size):
        h, w = image.shape[:2]
        new_h, new_w = size
        image = cv2.resize(image, (new_w, new_h))
        scale_x = new_w / w
        scale_y = new_h / h
        resized_bboxes = []
        for bbox in bboxes:
            x_min, y_min, w, h = bbox
            x_max = x_min+w
            y_max = y_min+h
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)
            resized_bboxes.append([x_min, y_min, x_max, y_max])
        return image, resized_bboxes

    def adjust_bboxes(self, bboxes, x_offset, y_offset):
        adjusted_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            adjusted_bboxes.append([
                x_min + x_offset, y_min + y_offset,
                x_max + x_offset, y_max + y_offset
            ])
        return adjusted_bboxes

    def mosaic_augmentation(self, images, bbox_list):
        h, w = self.output_size
        mosaic_image = np.zeros((h, w), dtype=np.uint8)
        all_bboxes = []
        resized_images = []
        resized_bboxes_list = []
        for img, bboxes in zip(images, bbox_list):
            resized_img, resized_bboxes = self.resize_image_and_bboxes(img, bboxes, (h // 2, w // 2))
            resized_images.append(resized_img)
            resized_bboxes_list.append(resized_bboxes)

        mosaic_image[:h // 2, :w // 2] = resized_images[0]
        all_bboxes.extend(self.adjust_bboxes(resized_bboxes_list[0], 0, 0))

        mosaic_image[:h // 2, w // 2:] = resized_images[1]
        all_bboxes.extend(self.adjust_bboxes(resized_bboxes_list[1], w // 2, 0))

        mosaic_image[h // 2:, :w // 2] = resized_images[2]
        all_bboxes.extend(self.adjust_bboxes(resized_bboxes_list[2], 0, h // 2))

        mosaic_image[h // 2:, w // 2:] = resized_images[3]
        all_bboxes.extend(self.adjust_bboxes(resized_bboxes_list[3], w // 2, h // 2))

        return mosaic_image, all_bboxes