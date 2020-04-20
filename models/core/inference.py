from torch.nn import Module

from .boxes import pred_loc_converter

class InferenceBox(Module):
    def __init__(self, conf_threshold=0.01, iou_threshold=0.45, topk=200):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk

    def forward(self, predicts, dboxes):
        """
        :param predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_nums)
        :param dboxes: Tensor, default boxes Tensor whose shape is (total_dbox_nums, 4)`
        :return:
        """

        """
        pred_loc, inf_loc: shape = (batch number, default boxes number, 4)
        pred_conf: shape = (batch number, default boxes number, class number)
        """
        pred_loc, pred_conf = predicts[:, :, :4], predicts[:, :, 4:]

        batch_num = predicts.shape[0]
        class_num = pred_conf.shape[2]

        inf_loc = pred_loc_converter(pred_loc, dboxes)
        indicator = pred_conf > self.conf_threshold

        for b in range(batch_num):
            pass

def non_maximum_suppression():
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
    """

def toVisualizeImg():
    pass
