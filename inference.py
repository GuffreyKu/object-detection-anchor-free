import torch
import cv2
import pickle
import random
from utils.tool import read_imgTotensor, predict_full, draw_bbox, load_annotation
from utils.pytorchtools import get_device, load_weights
from model.centerNet import CenterNet

DEVICE = get_device()

valid_path = "data/valid.pkl"
annotation_path = "data/train_dataset/train_label.json"
model_path = "savemodel/model.pth"
# Must match the backbone that wrote model_path, or load_weights says so by name.
backbone = "swin_t"

image_size = (512, 512)
conf = 0.1
nms_thres = 0.45

if __name__ == "__main__":
    _, names = load_annotation(annotation_path)
    class_names = dict(enumerate(names))
    # Distinct-ish colour per class without hand-listing 34 of them.
    color_map = {i: (37 * i % 256, 91 * i % 256, 173 * i % 256) for i in range(len(names))}

    with open(valid_path, 'rb') as file:
        valid_annotation = pickle.load(file)

    annotation_item = random.choice(valid_annotation)

    image_path = annotation_item["path"]
    print(image_path)
    print(annotation_item["bbox"], annotation_item["labels"])

    image, input_data = read_imgTotensor(image_path, image_size)

    # State dict, not the traced model: tracing captures only the rois=None branch,
    # so a traced model cannot run the second stage.
    model = CenterNet(num_classes=len(names), backbone=backbone).to(DEVICE)
    load_weights(model, torch.load(model_path, map_location=DEVICE), model_path)
    model.eval()
    with torch.no_grad():
        outputs = predict_full(model, input_data, image_size, conf, nms_thres, DEVICE)

    if len(outputs) > 0:
        outputs = outputs.data.cpu().numpy()
        labels = outputs[:, 5]
        scores = outputs[:, 4]
        bboxes = outputs[:, :4]

        # read_imgTotensor hands back RGB, cv2.imwrite wants BGR.
        image = draw_bbox(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                          bboxes, labels, class_names, color_map, scores=scores, show_name=True)
        cv2.imwrite("data/test.png", image)
    else:
        print(" nothing!! ")
