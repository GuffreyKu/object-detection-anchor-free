import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from .dataUtils import (Augment, Mosaic, encode_targets, aug_retangle, to_chw_tensor,
                        read_image_rgb)

def adjust_contrast(image, alpha, beta):
    # New image with adjusted contrast: new_image = alpha*image + beta
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

class ImgDataset(Dataset):
    def __init__(self, annotation, input_shape, num_classes, is_train, repeat=1, cache=False,
                 mosaic_p=0.2, cutout_p=0.2, sample_weights=None, gauss_min_overlap=0.7):
        super().__init__()
        '''
        annotation: [{"path": local image path,
                      "bbox": [[x1, y1, x2, y2], ...],
                      "labels": [category_id, ...]}]   as produced by utils.tool.load_annotation
        input_shape: (w, h)
        num_classes: number of cls in data
        is_train: if True use augmention
        repeat: virtual copies per image, each one gets freshly randomised augmentation
        cache: decode every image once at startup and keep it in RAM (see build_cache)
        '''
        # Must match the decoder's output stride in model/centerNet.py.
        self.stride = 4
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.num_classes = num_classes
        self.is_train = is_train
        self.repeat = repeat if is_train else 1

        self.annotation = [a for a in annotation if len(a["bbox"]) > 0]

        # Mosaic pastes four whole images into fixed quadrants at a hard 0.5x, with no
        # random centre and no per-tile scale, so every object in a mosaic sample is
        # exactly half size. A mosaic sample also carries ~4x the boxes (8.51 vs 2.13),
        # so the box-level share is much higher than the sample-level probability:
        # at the old p=0.5 that was 0.5*8.51/(0.5*8.51+0.5*2.13) = 80% of all training
        # boxes at half scale, pushing "min side < 16px" from 5.9% to 23.5% on a dataset
        # whose worst class already has a 30.8px median side. p=0.2 is the value that
        # makes the box-level split 50/50, which is why it is 0.2 and not 0.25.
        self.mosaic_p = mosaic_p if is_train else 0.0
        self.cutout_p = cutout_p if is_train else 0.0
        # Per-image sampling weights, shared with the DataLoader's sampler so that the
        # mosaic partner draw does not dilute it. None = uniform.
        self.sample_weights = sample_weights
        # Peak width of the heatmap target. Smaller (higher min_overlap) is a tighter,
        # more precise target but a weaker early gradient, since fewer cells carry any
        # signal at all - so it trades early learning speed for final peak sharpness.
        self.gauss_min_overlap = gauss_min_overlap

        if is_train:
            self.aug_fn = Augment()
            self.aug_mosaic = Mosaic(output_size=input_shape)

        self.cache_images = None
        self.cache_bboxes = None
        if cache:
            self.build_cache()

    def build_cache(self, workers=16):
        """
        Decode every image once into RAM, already resized to input_shape.

        Resized and not at source resolution because the source decodes to 118 GB
        (15127 images, median 1108x1477) against 11.9 GB at 512x512. The resize is
        deterministic and happened on every epoch anyway, so freezing it changes
        nothing - except inside Mosaic, which downsamples its four inputs to half the
        output size and now does that from 512 instead of from the original, i.e. one
        extra resampling step. Slightly softer mosaic tiles, no change to the geometry.

        One contiguous array, not a list of 15127 arrays: the DataLoader forks its
        workers, and pages stay shared copy-on-write only as long as nothing writes to
        them. Python's refcounter writes to every object header it touches, so a list
        would dirty a page per image per worker; with a single array it touches one
        object and the 11.9 GB behind it stays shared.
        """
        w, h = self.input_shape
        n = len(self.annotation)
        gb = n * h * w * 3 / 1e9
        self.cache_images = np.empty((n, h, w, 3), np.uint8)
        self.cache_bboxes = [None] * n

        def load(i):
            item = self.annotation[i]
            image = read_image_rgb(item["path"])
            raw_h, raw_w = image.shape[:2]
            self.cache_images[i] = cv2.resize(image, self.input_shape)
            # Boxes have to move into the resized frame with the pixels, because
            # load_sample no longer has the original dimensions to scale them by.
            rw, rh = w / raw_w, h / raw_h
            self.cache_bboxes[i] = [[b[0] * rw, b[1] * rh, b[2] * rw, b[3] * rh, lab]
                                    for b, lab in zip(item["bbox"], item["labels"])]

        # Threads, not processes: cv2 decode and resize release the GIL, and a process
        # pool would have to ship 11.9 GB back to the parent.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(load, range(n)), total=n,
                      desc=f"caching {'train' if self.is_train else 'valid'} {gb:.1f}GB",
                      ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'))

    def load_sample(self, index):
        '''Read one image. Returns RGB uint8 and bboxes as [x1, y1, x2, y2, label].'''
        i = index % len(self.annotation)

        if self.cache_images is not None:
            # Copy. Everything downstream is free to write into the array it is handed,
            # and the cache has to survive being served again next epoch.
            image = self.cache_images[i].copy()
            bboxes = [list(b) for b in self.cache_bboxes[i]]
        else:
            item = self.annotation[i]
            image = read_image_rgb(item["path"])
            bboxes = [list(box) + [label] for box, label in zip(item["bbox"], item["labels"])]

        if self.is_train and len(bboxes) > 2 and random.random() < self.cutout_p:
            # Cutout: paint over one object and drop its label with it. Fires here, i.e.
            # inside load_sample, so a mosaic sample rolls it once per tile - at the old
            # p=0.5 that erased an object from 93.75% of mosaic samples. num_mask is 1
            # rather than randint(1, n-1) for the same reason.
            image, bboxes = aug_retangle(image, bboxes, 1)

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

    def close_augment(self):
        """
        Turn off mosaic and cutout for the rest of training.

        Called once the LR has annealed far enough that the model is calibrating rather
        than exploring. Both of these distort what it is calibrating against: mosaic
        halves every object, and cutout erases objects together with their labels, i.e.
        teaches false negatives. Closing mosaic late is the standard YOLO recipe; the
        remaining photometric and affine augmentation stays on.
        """
        self.mosaic_p = 0.0
        self.cutout_p = 0.0

    def __getitem__(self, index):
        if self.is_train and random.random() < self.mosaic_p:
            # `index` is tile 0 rather than being discarded. The old version drew all
            # four tiles uniformly at random, which threw away whatever the sampler
            # chose - so a weighted sampler had no effect on mosaic samples at all.
            # The three partners are drawn from the same weights for the same reason.
            partners = (random.choices(range(len(self.annotation)), weights=self.sample_weights, k=3)
                        if self.sample_weights is not None
                        else random.sample(range(len(self.annotation)), 3))
            samples = [self.load_sample(i) for i in [index, *partners]]
            image, bboxes = self.aug_mosaic.mosaic_augmentation(
                images=[s[0] for s in samples], bbox_list=[s[1] for s in samples])
        else:
            image, bboxes = self.load_sample(index)

        # Cached images and their boxes are already in the input_shape frame, and the
        # mosaic branch composes straight to input_shape either way.
        if self.cache_images is None:
            image, bboxes = self.resize_image(image, bboxes)
        bboxes, labels = self.bbox_check(bboxes)

        if self.is_train:
            # Augment expects uint8 here, normalising first would make the HSV and
            # blur operators meaningless.
            image, bboxes = self.aug_fn(image, [list(b) + [l] for b, l in zip(bboxes, labels)])
            bboxes, labels = self.bbox_check(bboxes)

        heat_map, wh, offset, offset_mask = encode_targets(
            bboxes, labels, self.output_shape, self.num_classes, self.stride,
            self.gauss_min_overlap)

        # Raw boxes go out too: the RoI head is trained on ground-truth boxes, and the
        # encoded targets cannot be inverted back to them (a cell collision drops one).
        boxes = np.array(bboxes, np.float32).reshape(-1, 4)
        labels = np.array(labels, np.int64).reshape(-1)

        return to_chw_tensor(image), heat_map, wh, offset, offset_mask, boxes, labels

    def __len__(self):
        return len(self.annotation) * self.repeat


class CropDataset(Dataset):
    """Crops original RGB images for the standalone stage-2 classifier."""

    def __init__(self, samples, train=False, size=224, expand=0.1):
        self.samples = list(samples)
        self.train = train
        self.size = size
        self.expand = expand

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import math
        from .dataUtils import crop_centered_or_letterbox

        sample = self.samples[index]
        image = read_image_rgb(sample["path"])
        crop = crop_centered_or_letterbox(
            image, sample["box"], self.size, self.expand)
        if self.train and random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1])
        if self.train:
            gamma = math.exp(random.uniform(-0.2, 0.2))
            crop = (255 * (crop.astype(np.float32) / 255) ** gamma).astype(np.uint8)
        return to_chw_tensor(crop), int(sample["label"])


class ProposalCropDataset(Dataset):
    """Flatten cached proposal records into classifier crops."""

    def __init__(self, records, size=224, expand=0.1):
        self.records = records
        self.size = size
        self.expand = expand
        self.index = [(ri, pi) for ri, record in enumerate(records)
                      for pi in range(len(record["proposals"]))]
        self._cached_path = None
        self._cached_image = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from .dataUtils import crop_centered_or_letterbox

        record_id, proposal_id = self.index[index]
        record = self.records[record_id]
        if record["path"] != self._cached_path:
            self._cached_path = record["path"]
            self._cached_image = read_image_rgb(record["path"])
        crop = crop_centered_or_letterbox(
            self._cached_image, record["proposals"][proposal_id][:4],
            self.size, self.expand)
        return to_chw_tensor(crop), record_id, proposal_id
