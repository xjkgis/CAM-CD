import numpy as np
import torch

class Evaluator(object):
    def __init__(self, num_class, device='cuda'):
        self.num_class = num_class
        self.device = torch.device(device)
        self.confusion_matrix = torch.zeros(
            (self.num_class,) * 2, dtype=torch.int64, device=self.device
        )

    def get_tp_fp_tn_fn(self):
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        total_pixels = self.confusion_matrix.sum()
        tn = total_pixels - (tp + fp + fn)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, _, _ = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp + 1e-8)  
        return precision.cpu().numpy()

    def Recall(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn + 1e-8)
        return recall.cpu().numpy()

    def F1(self):
        precision_np = self.Precision()
        recall_np = self.Recall()
        F1 = (2.0 * precision_np * recall_np) / (precision_np + recall_np + 1e-8)
        return np.nan_to_num(F1)

    def Intersection_over_Union(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fp + fn + 1e-8)
        return IoU.cpu().numpy()

    def OA(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()  
        OA = (tp + tn) / (tp + fp + tn + fn + 1e-8)
        return OA.cpu().numpy()  

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].to(torch.int64) + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        dev = self.confusion_matrix.device
        if gt_image.device != dev:
            gt_image = gt_image.to(dev, non_blocking=True)
        if pre_image.device != dev:
            pre_image = pre_image.to(dev, non_blocking=True)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix.zero_()
