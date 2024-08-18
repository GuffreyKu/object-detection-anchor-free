import torch
import torch.nn as nn

class DIoULoss(nn.Module):
    def __init__(self):
        super(DIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        '''
        pred_boxes = [[x1, y1, x2, y2], [x1, y1, x2, y2], ....]
        target_boxes = [[x1, y1, x2, y2], [x1, y1, x2, y2], ....]
        '''
        num_preds = pred_boxes.size(0)
        num_targets = target_boxes.size(0)

        if num_preds != num_targets:
            # Calculate pairwise IoU between all predicted and target boxes
            pairwise_ious = torch.zeros((num_preds, num_targets), dtype=torch.float32)

            for i in range(num_preds):
                for j in range(num_targets):
                    pred = pred_boxes[i]
                    target = target_boxes[j]

                    pred_left_top = pred[:2] 
                    pred_right_bottom = pred[2:]
                    target_left_top = target[:2]
                    target_right_bottom = target[2:]

                    intersection_left_top = torch.max(pred_left_top, target_left_top)
                    intersection_right_bottom = torch.min(pred_right_bottom, target_right_bottom)
                    intersection_wh = torch.clamp(intersection_right_bottom - intersection_left_top, min=0)
                    intersection_area = intersection_wh[0] * intersection_wh[1]

                    target_w, target_h = target[2:] - target[:2]
                    pred_w, pred_h = pred[2:] - pred[:2]
                    
                    pred_area = pred_w * pred_h
                    target_area = target_w * target_h
                    union_area = pred_area + target_area - intersection_area

                    iou = intersection_area / (union_area + 1e-7)
                    pairwise_ious[i, j] = iou

            # Match each predicted box to the target box with the highest IoU
            ious, indices = pairwise_ious.max(dim=1)
            target_boxes = target_boxes[indices]

        # Now calculate the DIoU loss with matched boxes
        pred_centers = pred_boxes[:, :2]
        pred_sizes = pred_boxes[:, 2:]
        target_centers = target_boxes[:, :2]
        target_sizes = target_boxes[:, 2:]

        # Compute the IoU
        pred_left_top = pred_centers - pred_sizes / 2
        pred_right_bottom = pred_centers + pred_sizes / 2
        target_left_top = target_centers - target_sizes / 2
        target_right_bottom = target_centers + target_sizes / 2

        intersection_left_top = torch.max(pred_left_top, target_left_top)
        intersection_right_bottom = torch.min(pred_right_bottom, target_right_bottom)
        intersection_wh = torch.clamp(intersection_right_bottom - intersection_left_top, min=0)
        intersection_area = intersection_wh[:, 0] * intersection_wh[:, 1]

        pred_area = pred_sizes[:, 0] * pred_sizes[:, 1]
        target_area = target_sizes[:, 0] * target_sizes[:, 1]
        union_area = pred_area + target_area - intersection_area

        iou = intersection_area / (union_area + 1e-7)

        # Compute the Euclidean distance between the center points
        center_distance = torch.sum((pred_centers - target_centers) ** 2, dim=1)

        # Compute the diagonal length of the smallest enclosing box
        enclosing_left_top = torch.min(pred_left_top, target_left_top)
        enclosing_right_bottom = torch.max(pred_right_bottom, target_right_bottom)
        enclosing_wh = enclosing_right_bottom - enclosing_left_top
        enclosing_diagonal = torch.sum(enclosing_wh ** 2, dim=1)

        diou = iou - (center_distance / (enclosing_diagonal + 1e-7))

        return 1 - diou.mean()

if __name__ == "__main__":
    # Example usage
    pred_boxes = torch.tensor([[100, 100, 70, 70], [200, 200, 50, 50], [0, 0, 50, 50]], dtype=torch.float32)
    target_boxes = torch.tensor([[105, 105, 50, 50], [205, 205, 40, 40]], dtype=torch.float32)

    diou_loss = DIoULoss()
    loss = diou_loss(pred_boxes, target_boxes)
    print("DIoU Loss:", loss.item())
