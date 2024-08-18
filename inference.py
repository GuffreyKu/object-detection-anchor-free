import torch
import cv2
import pickle
import random
import matplotlib.pyplot as plt
from utils.tool import read_imgTotensor, predict, draw_bbox

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_path = "data/train.pkl"
valid_path = "data/valid.pkl"

image_size = (512, 512)
conf = 0.1
nms_thres = 0.45

color_map = {
    0:(0, 0, 255),
    1:(0, 255, 0)
    }

class_names = {
            0:"dog",
            1:"cat"
        }

model = torch.jit.load("savemodel/model_trace.pt")

def resize_bbox(image_path, bbox, input_shape):
    image = cv2.imread(image_path)
    raw_h, raw_w, _ = image.shape
    
    ratio_w = input_shape[0]/raw_w
    ratio_h = input_shape[1]/raw_h

    resize_bbox = []
    
    nx = int(bbox[0] * ratio_w)
    ny = int(bbox[1] * ratio_h) 
    nw = int(bbox[2] * ratio_w) 
    nh = int(bbox[3] * ratio_h) 
    
    resize_bbox.append([nx, ny, nx+nw, ny+nh])

    return resize_bbox

if __name__ == "__main__":
    data_path = "data/v1/"
    with open(train_path, 'rb') as file:
        valid_annotation = pickle.load(file)

    annotation_item = random.sample(valid_annotation, 1)
    
    print(annotation_item)
    image_path = annotation_item[0]["path"]
    name = image_path.split('/')[-1]
    image_path = data_path+name
    print(image_path)
    categories = annotation_item[0]["bbox"]
    print(categories)
    
    image, input_data = read_imgTotensor(image_path, image_size)
    
    model.eval()
    with torch.no_grad():
        outputs = predict(input_data, image_size, conf, nms_thres, model, DEVICE)

    if len(outputs) > 0:
        outputs = outputs.data.cpu().numpy()
        labels = outputs[:, 5]
        scores = outputs[:, 4]
        bboxes = outputs[:, :4]

        image = draw_bbox(image, bboxes, labels, class_names, color_map, scores=scores, show_name=True)
        cv2.imwrite("data/test.png", image)
    else:
        print(" nothing!! ")