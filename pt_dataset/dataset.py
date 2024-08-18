import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import cv2
import boto3
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from dotenv import load_dotenv
from .dataUtils import ImgAugTransform, Mosaic, gaussian_radius, draw_gaussian, aug_retangle, aug_circle, drop_cap_bbox

def minio_client_fn():
    load_dotenv(".env")
    endpoint = os.getenv("ENDPOINT")
    access = os.getenv("ACCESS")
    secret = os.getenv("SECRET")
    minio_client = boto3.client(
                        's3',
                        endpoint_url=endpoint,
                        aws_access_key_id=access,
                        aws_secret_access_key=secret
                    )
    return minio_client
        
def download_image(sub_path, data_path = "data/v2/"):
    name = sub_path.split('/')[-1]
    bucket_name = sub_path.split('/')[2]
    file_key = '/'.join(sub_path.split('/')[3:])

    image_path = data_path+name

    if not os.path.exists(image_path):
        minio_client = minio_client_fn()
        minio_client.download_file(bucket_name, file_key, image_path)
    return image_path

def adjust_contrast(image, alpha, beta):
    # New image with adjusted contrast: new_image = alpha*image + beta
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

class ImgDataset(Dataset):
    def __init__(self, annotation, input_shape, num_classes, is_train):
        super().__init__()
        '''
        annotation: [{ "path":xxx.bmp, "categories": {"name":[x1, y1, w, h]}}]
        input_shape: (w, h)
        num_classes: number of cls in data
        is_train: if True use augmention
        '''
        self.data_path = "data/v2/"
        self.stride = 2
        self.annotation = annotation
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.num_classes = num_classes
        self.is_train = is_train
        self.epoch = 0

        if is_train:
            self.aug_fn = ImgAugTransform()
            self.aug_mosaic = Mosaic(output_size=input_shape)

        self.__read_all()

    def data_prep(self, input, mode="div"):
        if mode == "div":
            input = input/255
            input = input.astype(np.float32)
            return input
        else:
            raise " ~ error prep mode"
        
    def resize_image(self, image, bboxes):
        raw_h, raw_w= image.shape
        image = cv2.resize(image, self.input_shape)
        image = self.data_prep(image, "div")

        ratio_w = self.input_shape[0]/raw_w
        ratio_h = self.input_shape[1]/raw_h

        resize_bbox = []
        for bbox in bboxes:
            nx = bbox[0] * ratio_w 
            ny = bbox[1] * ratio_h 
            nw = bbox[2] * ratio_w 
            nh = bbox[3] * ratio_h 
            resize_bbox.append([nx, ny, nx+nw, ny+nh, 0]) # 0 is "text", its mean find all of word from image

        return image, resize_bbox

    
    def bbox_check(self, bboxes):
        clip_bboxes = []
        labels = []

        for bbox in bboxes:
            x1, y1, x2, y2, label = bbox
            if x2 <= x1 or y2 <= y1:
                # Don't use such boxes as this may cause nan loss.
                continue
            x1 = int(np.clip(x1, 0, self.input_shape[1]))
            y1 = int(np.clip(y1, 0, self.input_shape[0]))
            x2 = int(np.clip(x2, 0, self.input_shape[1]))
            y2 = int(np.clip(y2, 0, self.input_shape[0]))

            # Clipping coordinates between 0 to image dimensions as negative values
            # or values greater than image dimensions may cause nan loss.
            clip_bboxes.append([x1, y1, x2, y2])
            labels.append(label)
        return clip_bboxes, labels
    
    def __read_all(self):
        self.images = []
        self.bboxes = []
        data_size = len(self.annotation)

        for idx in range(data_size):
            sub_annotation = self.annotation[idx]
            image_path = download_image(sub_path=sub_annotation["path"],
                                        data_path=self.data_path)
            bboxes = sub_annotation["bbox"]
            texts = sub_annotation["transcription"]
            raw_bboxes = drop_cap_bbox(bboxes, texts)
            raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
 
            self.images.append(raw_image)
            self.bboxes.append(raw_bboxes)
            
            # 1 image generator k image, k=10
            for _ in range(10):
                num_mask = random.randint(1, len(raw_bboxes)-1)
                image_ra, bboxes_ra = aug_retangle(raw_image, raw_bboxes, num_mask)

                num_mask = random.randint(1, len(raw_bboxes)-1)
                image_cc, bboxes_cc = aug_circle(raw_image, raw_bboxes, num_mask)

                if len(bboxes_ra) > 1 :
                    self.images.append(image_ra)
                    self.bboxes.append(bboxes_ra)

                if len(bboxes_cc) > 1 :
                    self.images.append(image_cc)
                    self.bboxes.append(bboxes_cc)

    def get_number_data(self):
        return len(self.images)
    
    def __getitem__(self, index):
        heat_map = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        offset = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        offset_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        if (random.randint(0, 1)) and (self.is_train):
            idx_sample = random.sample(range(len(self.images)), 4)
            mosaic_images = [self.images[idx] for idx in idx_sample]
            mosaic_bboxes = [self.bboxes[idx] for idx in idx_sample]
            image, bboxes = self.aug_mosaic.mosaic_augmentation(images=mosaic_images, bbox_list=mosaic_bboxes)
        else:
            image = self.images[index]
            bboxes = self.bboxes[index]

        image, bboxes = self.resize_image(image, bboxes)
        bboxes, labels = self.bbox_check(bboxes)

        if self.is_train:
            image, bboxes = self.aug_fn(image, bboxes)
        
        image = np.expand_dims(image, axis=0) 
        bboxes = np.array(bboxes)
        labels = np.array(labels)

        if bboxes.size == 0:
            print(bboxes)
            raise ValueError("bbox error")

        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] / self.stride, a_min=0, a_max=self.output_shape[1])
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] / self.stride, a_min=0, a_max=self.output_shape[0])

        for i in range(len(labels)):
            x1, y1, x2, y2 = bboxes[i]
            cls_id = int(labels[i])
            h, w = y2 - y1, x2 - x1
            if ( h > 0) and (w > 0): # 
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # Calculates the feature points of the real box
                ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                # Get gaussian heat map
                heat_map[:, :, cls_id] = draw_gaussian(heat_map[:, :, cls_id], ct_int, radius)
                # Assign ground truth height and width
                wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                # Assign center point offset
                offset[ct_int[1], ct_int[0]] = ct - ct_int
                # Set the corresponding mask to 1
                offset_mask[ct_int[1], ct_int[0]] = 1

        image = image.astype(np.float32)

        return image, heat_map, wh, offset, offset_mask
    
    def __len__(self):
        self.epoch += 1
        return len(self.images)