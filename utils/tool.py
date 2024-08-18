import os
import cv2
import json
import random
import torch
import pickle
import numpy as np
from utils.detect import postprocess_output, decode_bbox
from model.centerNet import CenterNetPoolingNMS

def folderCheck(folders:list):
    for path in folders:
        if not os.path.exists(path):
            os.mkdir(path)


def load_annotation(file_path):
    '''
    file_path : json file path (format label studio mini json)

    return 
    [ 
        { 
          "path": minio url,
          "transcription" : [text1, text2, ....]
          "bbox" : [[text1_bbox], [text2_bbox]]
        },
        {
        ....next image
        }
    ]
    '''

    final = []

    with open(file_path, 'r') as file:
        data = json.load(file)

    for i in range(len(data)):
        sub_item = {}
        path = data[i]['ocr']
        if "label" in data[i]:
            tar_bbox = data[i]['label']
            bboxes = []
            for ind in range(len(tar_bbox)):
                ow = tar_bbox[ind]["original_width"]
                oh = tar_bbox[ind]["original_height"]

                x = tar_bbox[ind]["x"] * (ow/100)
                y = tar_bbox[ind]["y"] * (oh/100)
                w = tar_bbox[ind]["width"] * (ow/100)
                h = tar_bbox[ind]["height"] * (oh/100)
                bboxes.append([x, y, w, h])

            sub_item["path"] = path
            sub_item["transcription"] = data[i]['transcription']
            sub_item["bbox"] = bboxes
            if len(sub_item["bbox"]) > 1:
                final.append(sub_item)

    return final

def train_test_split(annotations, save_path="data/"):
    random.shuffle(annotations)

    split_index = int(len(annotations) * 0.8)

    train_annotation = annotations[:split_index]
    valid_annotation = annotations[split_index:]

    with open(save_path+'train.pkl', 'wb') as file:
        pickle.dump(train_annotation, file)

    with open(save_path+'valid.pkl', 'wb') as file:
        pickle.dump(valid_annotation, file)

    return train_annotation, valid_annotation


def read_imgTotensor(path, image_size):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, image_size)

    input_data = np.expand_dims(image, 0)
    input_data = np.expand_dims(input_data, 0)
    input_data = torch.from_numpy(input_data).float()
    input_data /= 255

    return image, input_data

def predict(input_data, image_size, conf, nms_thres, model, dev):
    """
    Predict one image
    Args:
        image: input image
        model: CenterNet model
        dev: torch device
        args: ArgumentParser

    Returns:  bounding box of one image(x1, y1, x2, y2 score, label).

    """
    
    input_data = input_data.to(dev)

    hms, whs, offsets = model(input_data)
    hms = CenterNetPoolingNMS(kernel=3)(hms)

    hms = hms.permute(0, 2, 3, 1)
    whs = whs.permute(0, 2, 3, 1)
    offsets = offsets.permute(0, 2, 3, 1)

    outputs = postprocess_output(hms, whs, offsets, conf, dev)
    outputs = decode_bbox(outputs,
                          image_size,
                          dev, image_shape=image_size, remove_pad=True,
                          need_nms=True, nms_thres=nms_thres)

    return outputs[0]

def decoder(prediction, image_size, conf, nms_thres, dev):
    """
    Predict one image
    Args:
        image: input image
        model: CenterNet model
        dev: torch device
        args: ArgumentParser

    Returns:  bounding box of one image(x1, y1, x2, y2 score, label).

    """

    kernel = 3
    pad = (kernel-1)//2
    max_hms = torch.nn.functional.max_pool2d(prediction["hms"], kernel_size=kernel, stride=1, padding=pad)
    keep = (max_hms == prediction["hms"]).float()
    prediction["hms"] *= keep

    outputs = postprocess_output(prediction["hms"], prediction["whs"], prediction["offsets"], conf, dev)
    outputs = decode_bbox(outputs,
                          image_size,
                          dev, image_shape=image_size, remove_pad=True,
                          need_nms=True, nms_thres=nms_thres)

    return outputs[0]

def draw_bbox(image, bboxes, labels, class_names, color_map, scores=None, show_name=False):
    """
    Draw bounding box in image.
    Args:
        image: image
        bboxes: coordinate of bounding box
        labels: the index of labels
        class_names: the names of class
        scores: bounding box confidence
        show_name: show class name if set true, otherwise show index of class

    Returns: draw result

    """
    
    image_height, image_width = image.shape[:2]
    draw_image = image.copy()

    for i, c in list(enumerate(labels)):
        bbox = bboxes[i]
        c = int(c)
        color = [int(j) for j in color_map[c]]
        if show_name:
            predicted_class = class_names[c]
        else:
            predicted_class = c

        if scores is None:
            text = '{}'.format(predicted_class)
        else:
            score = scores[i]
            text = '{} {:.2f}'.format(predicted_class, score)

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 -y1

        x1 = max(0, np.floor(x1).astype(np.int32))
        y1 = max(0, np.floor(y1).astype(np.int32))
        x2 = min(image_width, np.floor(x2).astype(np.int32))
        y2 = min(image_height, np.floor(y2).astype(np.int32))

        thickness = int((image_height + image_width) / (np.sqrt(image_height**2 + image_width**2)))
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), color=color, thickness=thickness)


    return draw_image