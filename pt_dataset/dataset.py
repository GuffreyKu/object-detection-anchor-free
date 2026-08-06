import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from .dataUtils import (Augment, Mosaic, encode_targets, aug_retangle, to_chw_tensor,
                        read_image_rgb)

def adjust_contrast(image, alpha, beta):
    # New image with adjusted contrast: new_image = alpha*image + beta
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

class ImgDataset(Dataset):
    def __init__(self, annotation, input_shape, num_classes, is_train, repeat=1):
        super().__init__()
        '''
        annotation: [{"path": local image path,
                      "bbox": [[x1, y1, x2, y2], ...],
                      "labels": [category_id, ...]}]   as produced by utils.tool.load_annotation
        input_shape: (w, h)
        num_classes: number of cls in data
        is_train: if True use augmention
        repeat: virtual copies per image, each one gets freshly randomised augmentation
        '''
        # Must match the decoder's output stride in model/centerNet.py.
        self.stride = 4
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.num_classes = num_classes
        self.is_train = is_train
        self.repeat = repeat if is_train else 1

        self.annotation = [a for a in annotation if len(a["bbox"]) > 0]

        if is_train:
            self.aug_fn = Augment()
            self.aug_mosaic = Mosaic(output_size=input_shape)

    def load_sample(self, index):
        '''Read one image from disk. Returns RGB uint8 and bboxes as [x1, y1, x2, y2, label].'''
        item = self.annotation[index % len(self.annotation)]
        image = read_image_rgb(item["path"])

        bboxes = [list(box) + [label] for box, label in zip(item["bbox"], item["labels"])]

        if self.is_train and len(bboxes) > 2 and random.random() < 0.5:
            # Cutout: paint over some objects and drop their labels with them.
            image, bboxes = aug_retangle(image, bboxes, random.randint(1, len(bboxes) - 1))

        return image, bboxes

    def resize_image(self, image, bboxes):
        '''Resize to input_shape, keeping uint8. bboxes in and out are [x1, y1, x2, y2, label].'''
        raw_h, raw_w = image.shape[:2]
        image = cv2.resize(image, self.input_shape)

        ratio_w = self.input_shape[0]/raw_w
        ratio_h = self.input_shape[1]/raw_h

        resize_bbox = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            resize_bbox.append([x1 * ratio_w, y1 * ratio_h, x2 * ratio_w, y2 * ratio_h, bbox[4]])

        return image, resize_bbox

    def bbox_check(self, bboxes):
        clip_bboxes = []
        labels = []
        in_w, in_h = self.input_shape
        # One heatmap cell. Anything thinner cannot be encoded anyway, and after the
        # affine augmentation it is usually a sliver of an object pushed off-frame,
        # which would teach the model that a bottle edge is a bottle.
        min_size = self.stride

        for bbox in bboxes:
            x1, y1, x2, y2, label = bbox
            if x2 <= x1 or y2 <= y1:
                # Don't use such boxes as this may cause nan loss.
                continue
            # Clipping coordinates between 0 to image dimensions as negative values
            # or values greater than image dimensions may cause nan loss.
            # Kept as floats so the offset head still sees the sub-pixel remainder.
            x1 = float(np.clip(x1, 0, in_w))
            y1 = float(np.clip(y1, 0, in_h))
            x2 = float(np.clip(x2, 0, in_w))
            y2 = float(np.clip(y2, 0, in_h))
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue

            clip_bboxes.append([x1, y1, x2, y2])
            labels.append(label)
        return clip_bboxes, labels

    def get_number_data(self):
        return len(self)

    def __getitem__(self, index):
        if self.is_train and random.randint(0, 1):
            idx_sample = random.sample(range(len(self.annotation)), 4)
            samples = [self.load_sample(idx) for idx in idx_sample]
            image, bboxes = self.aug_mosaic.mosaic_augmentation(
                images=[s[0] for s in samples], bbox_list=[s[1] for s in samples])
        else:
            image, bboxes = self.load_sample(index)

        image, bboxes = self.resize_image(image, bboxes)
        bboxes, labels = self.bbox_check(bboxes)

        if self.is_train:
            # Augment expects uint8 here, normalising first would make the HSV and
            # blur operators meaningless.
            image, bboxes = self.aug_fn(image, [list(b) + [l] for b, l in zip(bboxes, labels)])
            bboxes, labels = self.bbox_check(bboxes)

        heat_map, wh, offset, offset_mask = encode_targets(
            bboxes, labels, self.output_shape, self.num_classes, self.stride)

        # Raw boxes go out too: the RoI head is trained on ground-truth boxes, and the
        # encoded targets cannot be inverted back to them (a cell collision drops one).
        boxes = np.array(bboxes, np.float32).reshape(-1, 4)
        labels = np.array(labels, np.int64).reshape(-1)

        return to_chw_tensor(image), heat_map, wh, offset, offset_mask, boxes, labels

    def __len__(self):
        return len(self.annotation) * self.repeat
