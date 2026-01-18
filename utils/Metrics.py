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
        # 5. 在GPU上计算，只在最后返回时传输到CPU
        precision = tp / (tp + fp + 1e-8)  # 加epsilon防除零
        return precision.cpu().numpy()

    def Recall(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn + 1e-8)
        return recall.cpu().numpy()

    def F1(self):
        # F1的公式不变，输入来自已经转为numpy的Precision和Recall
        precision_np = self.Precision()
        recall_np = self.Recall()
        F1 = (2.0 * precision_np * recall_np) / (precision_np + recall_np + 1e-8)
        # F1可能为nan，转为0
        return np.nan_to_num(F1)

    def Intersection_over_Union(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fp + fn + 1e-8)
        return IoU.cpu().numpy()

    def OA(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()  # <--- 也确保这里调用的是带下划线的方法
        OA = (tp + tn) / (tp + fp + tn + fn + 1e-8)
        return OA.cpu().numpy()  # <--- 正确：返回Numpy数组

    def Kappa(self):
        # Kappa的torch实现可以更简洁，并且避免中间变量使用numpy
        total = self.confusion_matrix.sum()
        po = torch.diag(self.confusion_matrix).sum() / (total + 1e-8)
        pe_row = self.confusion_matrix.sum(dim=0)
        pe_col = self.confusion_matrix.sum(dim=1)
        pe = (pe_row * pe_col).sum() / (total * total + 1e-8)

        kappa = (po - pe) / (1 - pe + 1e-8)
        return kappa.cpu().numpy()  # <--- 正确：返回Numpy标量

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].to(torch.int64) + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # 对齐输入到内部矩阵所在设备，避免设备不一致
        dev = self.confusion_matrix.device
        if gt_image.device != dev:
            gt_image = gt_image.to(dev, non_blocking=True)
        if pre_image.device != dev:
            pre_image = pre_image.to(dev, non_blocking=True)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix.zero_()


if __name__ == "__main__":
    # --- 1. 确定测试设备 ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- Testing Evaluator on device: {device} ---")

    # --- 2. 创建评估器实例，并指定设备 ---
    num_class = 2
    evaluator = Evaluator(num_class, device=device)

    # --- 3. 构造模拟数据，并将其转换为指定设备上的Tensor ---
    gt_np = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    pred_np = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    gt_labels = torch.from_numpy(gt_np).to(device)
    pred_labels = torch.from_numpy(pred_np).to(device)

    # --- 4. 将GPU张量添加到评估器中 ---
    evaluator.add_batch(gt_labels, pred_labels)

    # --- 5. 计算所有指标 ---
    oa_array = evaluator.OA()
    precision_array = evaluator.Precision()
    recall_array = evaluator.Recall()
    f1_array = evaluator.F1()
    iou_array = evaluator.Intersection_over_Union()
    kappa = evaluator.Kappa()

    # --- 6. 提取和打印结果 ---
    overall_accuracy = oa_array[0]
    precision_change = precision_array[1]
    recall_change = recall_array[1]
    f1_change = f1_array[1]
    iou_change = iou_array[1]

    print("================== 变化检测精度评定测试 (GPU优化类) ==================\n")
    # 打印混淆矩阵时，需要从GPU传回CPU
    print(f"混淆矩阵:\n{evaluator.confusion_matrix.cpu().numpy()}\n")
    print("--- 变化检测核心指标 (Core Metrics for Change Detection) ---\n")
    print(f"Overall Accuracy (OA):   {overall_accuracy:.4f}")
    print(f"Precision (for 'Change'):  {precision_change:.4f}")
    print(f"Recall (for 'Change'):     {recall_change:.4f}")
    print(f"F1-Score (for 'Change'):   {f1_change:.4f}")
    print(f"IoU (for 'Change'):        {iou_change:.4f}")
    print(f"Kappa:                     {kappa:.4f}\n")
    print("=====================================================================")
