import cv2
import math
import random
import numpy as np

# The backbone is ImageNet-pretrained, so the input has to arrive in the distribution
# those weights were fitted on. Plain /255 leaves it off by roughly one std per channel.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_image_rgb(path):
    """
    Read an image as RGB in the SAME coordinate frame the annotations use.

    IMREAD_IGNORE_ORIENTATION is the point of this function. 360 of the 15127 images
    carry an EXIF rotation and cv2 applies it by default, transposing the image out
    from under the boxes. Measured: without the flag only 69.3% of the 1005 boxes on
    those images still fall inside the frame, and the rest either crash the augmenter
    or get silently clipped onto the wrong pixels. With it, all 15127 decoded sizes
    match the json exactly and every box fits.

    Anything that reads a training image must go through here, or inference will see a
    different frame than training did.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_chw_tensor(image):
    """uint8 RGB HWC -> float32 CHW, ImageNet-normalised. Used by training and inference."""
    x = image.astype(np.float32) / 255.
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))

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

def gaussian_radius(det_size, min_overlap=0.7):
    # min_overlap is 0.7, the CornerNet/CenterNet reference value. It used to be 0.3,
    # which inflates every radius by ~1.81x: measured over all 32189 boxes at stride 4
    # the median radius was 10 cells and the max 63, i.e. one object's soft-negative
    # blob covering the whole 128x128 map. focal_loss only treats target==1 as positive
    # and down-weights the rest by (1-target)^4, so an oversized blob turns a 21x21
    # annulus per object into supervision that is neither positive nor a real negative:
    # flatter peaks, more survivors through the 3x3 peak_filter, and wh/offset read at
    # off-centre cells. At 0.7 the median is 5 and the max 34.
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


def encode_targets(bboxes, labels, output_shape, num_classes, stride, min_overlap=0.7):
    """
    Encode boxes into the CenterNet training targets.
    Args:
        bboxes: [[x1, y1, x2, y2], ...] in model input pixels
        labels: class index per box
        output_shape: (w, h) of the heatmap
        num_classes: number of classes
        stride: input pixels per heatmap cell

    Returns: heat_map, wh, offset, offset_mask

    """
    out_w, out_h = output_shape
    heat_map = np.zeros((out_h, out_w, num_classes), dtype=np.float32)
    wh = np.zeros((out_h, out_w, 2), dtype=np.float32)
    offset = np.zeros((out_h, out_w, 2), dtype=np.float32)
    offset_mask = np.zeros((out_h, out_w), dtype=np.float32)

    for bbox, label in zip(bboxes, labels):
        # Keep sub-pixel precision here, the offset head is trained on the remainder.
        x1, y1 = bbox[0] / stride, bbox[1] / stride
        x2, y2 = bbox[2] / stride, bbox[3] / stride
        x1, x2 = np.clip([x1, x2], 0, out_w - 1)
        y1, y2 = np.clip([y1, y2], 0, out_h - 1)

        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap)))
        ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        cls_id = int(label)

        draw_gaussian(heat_map[:, :, cls_id], ct_int, radius)
        wh[ct_int[1], ct_int[0]] = w, h
        offset[ct_int[1], ct_int[0]] = ct - ct_int
        offset_mask[ct_int[1], ct_int[0]] = 1

    return heat_map, wh, offset, offset_mask


def aug_retangle(image, bboxes, num_mask = 1):
    """Blank out num_mask boxes with a flat rectangle. bboxes are [x1, y1, x2, y2, ...].

    This deletes the labels along with the pixels, so it is a false-negative generator
    if it runs too often. Callers should keep num_mask at 1 and the probability low, and
    should turn it off entirely once the LR has annealed - see close_augment().
    """
    if num_mask >= len(bboxes):
        num_mask = 1
    # Copy: the caller keeps the unmasked image, and every augmented sample needs its own buffer.
    image = image.copy()
    mask_bbox = random.sample(bboxes, num_mask)
    h, w = image.shape[:2]
    for mask in mask_bbox:
        # Clip before indexing. Boxes are still in raw image coordinates here, and one
        # bad annotation must not take down a ten-hour training run.
        x1, y1, x2, y2 = (int(v) for v in mask[:4])
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        fill = np.atleast_1d(image[y1, x1]).tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), fill, thickness=-1)

    drop_mask_bboxes = [item for item in bboxes if item not in mask_bbox]
    return image, drop_mask_bboxes


class Augment:
    """Photometric and geometric augmentation on uint8 RGB. bboxes are [x1, y1, x2, y2, ...].

    Replaces the imgaug version, which applied exactly one randomly chosen operator per
    image and forced numpy<2. Here each operator fires on its own probability, so a
    sample can be flipped and colour-shifted and blurred, which is what actually widens
    the training distribution.
    """

    def __init__(self, hue=6, sat=0.3, gamma=0.35, scale=0.3, translate=0.1,
                 flip_p=0.5, blur_p=0.15, photo_p=0.8, affine_p=0.7):
        self.hue, self.sat, self.gamma = hue, sat, gamma
        self.scale, self.translate = scale, translate
        self.flip_p, self.blur_p = flip_p, blur_p
        self.photo_p, self.affine_p = photo_p, affine_p

    def hsv_jitter(self, image):
        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        # Hue stays on a deliberately short leash. Colour is the only cue separating
        # glass_bottle / metal_can / plastic_bottle (their aspect ratios all sit in
        # 0.94-1.11), so a wide hue shift would augment the label away.
        h = ((h.astype(np.int16) + random.randint(-self.hue, self.hue)) % 180).astype(np.uint8)
        s = np.clip(s * (1 + random.uniform(-self.sat, self.sat)), 0, 255).astype(np.uint8)
        # Brightness by gamma, not by multiply-and-clip. A specular highlight is the
        # strongest single material cue in this dataset (glass and plastic reflect the
        # illuminant colour, metal tints it, carton barely reflects at all); scaling v
        # up flattens every highlight into the same clipped 255 and erases that cue.
        # Gamma is monotonic, so it never clips and the highlight ordering survives.
        g = math.exp(random.uniform(-self.gamma, self.gamma))
        v = (255 * (v / 255) ** g).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

    def affine(self, image, bboxes):
        h, w = image.shape[:2]
        s = random.uniform(1 - self.scale, 1 + self.scale)
        tx = random.uniform(-self.translate, self.translate) * w
        ty = random.uniform(-self.translate, self.translate) * h
        # Scale about the image centre, then translate.
        m = np.float32([[s, 0, (1 - s) * w / 2 + tx],
                        [0, s, (1 - s) * h / 2 + ty]])
        image = cv2.warpAffine(image, m, (w, h), borderValue=(114, 114, 114))
        bboxes = [[s * b[0] + m[0, 2], s * b[1] + m[1, 2],
                   s * b[2] + m[0, 2], s * b[3] + m[1, 2], *b[4:]] for b in bboxes]
        return image, bboxes

    def __call__(self, image, bboxes):
        if random.random() < self.flip_p:
            w = image.shape[1]
            image = np.ascontiguousarray(image[:, ::-1])
            bboxes = [[w - b[2], b[1], w - b[0], b[3], *b[4:]] for b in bboxes]

        if random.random() < self.photo_p:
            image = self.hsv_jitter(image)

        if random.random() < self.blur_p:
            image = cv2.GaussianBlur(image, (5, 5), random.uniform(0.1, 1.5))

        if random.random() < self.affine_p:
            image, bboxes = self.affine(image, bboxes)

        # Boxes pushed off-frame come back clipped or dropped by ImgDataset.bbox_check.
        return image, bboxes


class Mosaic:
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_image_and_bboxes(self, image, bboxes, size):
        '''bboxes in and out are [x1, y1, x2, y2, ...], trailing fields (label) pass through.'''
        h, w = image.shape[:2]
        new_h, new_w = size
        image = cv2.resize(image, (new_w, new_h))
        scale_x = new_w / w
        scale_y = new_h / h
        resized_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            resized_bboxes.append([x_min * scale_x, y_min * scale_y,
                                   x_max * scale_x, y_max * scale_y, *bbox[4:]])
        return image, resized_bboxes

    def adjust_bboxes(self, bboxes, x_offset, y_offset):
        adjusted_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            adjusted_bboxes.append([
                x_min + x_offset, y_min + y_offset,
                x_max + x_offset, y_max + y_offset, *bbox[4:]
            ])
        return adjusted_bboxes

    def mosaic_augmentation(self, images, bbox_list):
        w, h = self.output_size
        mosaic_image = np.zeros((h, w, 3), dtype=np.uint8)
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


def crop_and_letterbox(image, box, size=224, expand=0.1, fill=114):
    """Expand an xyxy box, crop it, and preserve its aspect ratio in a square canvas."""
    raw_h, raw_w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid crop box: {box}")
    x1, x2 = max(0, math.floor(x1 - w * expand)), min(raw_w, math.ceil(x2 + w * expand))
    y1, y2 = max(0, math.floor(y1 - h * expand)), min(raw_h, math.ceil(y2 + h * expand))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"crop outside image: {box}")
    scale = min(size / crop.shape[1], size / crop.shape[0])
    new_w = max(1, round(crop.shape[1] * scale))
    new_h = max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), fill, np.uint8)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas
