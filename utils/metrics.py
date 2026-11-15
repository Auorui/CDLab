import torch
import torchmetrics
import prettytable

class CDMetrics:
    def __init__(self, num_classes: int, device='cuda:0', ignore_index=None):
        self.num_classes = num_classes
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if num_classes == 2:
            self.metric_acc = torchmetrics.Accuracy(task='binary', ignore_index=ignore_index).to(self.device)
            self.metric_prec = torchmetrics.Precision(task='binary', ignore_index=ignore_index).to(self.device)
            self.metric_recall = torchmetrics.Recall(task='binary', ignore_index=ignore_index).to(self.device)
            self.metric_f1 = torchmetrics.F1Score(task='binary', ignore_index=ignore_index).to(self.device)
            self.metric_iou = torchmetrics.JaccardIndex(task='binary', ignore_index=ignore_index).to(self.device)
        else:
            metric_cfg = dict(
                task='multiclass',
                num_classes=self.num_classes,
                average='macro',
                ignore_index=ignore_index
            )
            self.metric_acc = torchmetrics.Accuracy(
                task='multiclass', num_classes=self.num_classes, average='micro', ignore_index=ignore_index).to(self.device)
            self.metric_prec = torchmetrics.Precision(**metric_cfg).to(self.device)
            self.metric_recall = torchmetrics.Recall(**metric_cfg).to(self.device)
            self.metric_f1 = torchmetrics.F1Score(**metric_cfg).to(self.device)
            self.metric_iou = torchmetrics.JaccardIndex(**metric_cfg).to(self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds:  (B, H, W)
            target: (B, H, W)
        """
        self.metric_acc.update(preds, target)
        self.metric_prec.update(preds, target)
        self.metric_recall.update(preds, target)
        self.metric_f1.update(preds, target)
        self.metric_iou.update(preds, target)


    def compute(self):
        result = {
            'OA': self.metric_acc.compute().item(),
            'P': self.metric_prec.compute().item(),
            'R': self.metric_recall.compute().item(),
            'F1': self.metric_f1.compute().item(),
            'IoU': self.metric_iou.compute().item(),
        }
        return result

    def reset(self):
        self.metric_acc.reset()
        self.metric_prec.reset()
        self.metric_recall.reset()
        self.metric_f1.reset()
        self.metric_iou.reset()


def print_metrics_table(metrics_dict, epoch, train_loss=None, val_loss=None):
    """在控制台打印横向排列的美观评估指标表格"""
    if train_loss is not None and val_loss is not None:
        print(f"\nEpoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    table = prettytable.PrettyTable()

    # 直接使用简写作为表头
    field_names = ["OA", "P", "R", "F1", "IoU"]
    row_values = [
        f"{metrics_dict.get('OA', 0):.4f}",
        f"{metrics_dict.get('P', 0):.4f}",
        f"{metrics_dict.get('R', 0):.4f}",
        f"{metrics_dict.get('F1', 0):.4f}",
        f"{metrics_dict.get('IoU', 0):.4f}"
    ]

    table.field_names = field_names
    table.add_row(row_values)

    # 设置居中对齐
    table.align = "c"

    table.set_style(prettytable.MARKDOWN)
    print(table)


if __name__=="__main__":
    import cv2
    import numpy as np
    # import segmentation_models_pytorch as smp

    num_classes = 3
    device = "cuda"

    # ---- 读图 ----
    label_img = cv2.imread("label_3.png", cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread("pred_3.png", cv2.IMREAD_GRAYSCALE)

    print(f"原始 label shape: {label_img.shape}, 取值: {np.unique(label_img)}")
    print(f"原始 pred  shape: {pred_img.shape}, 取值: {np.unique(pred_img)}")

    # ---- 类别映射 ----
    if num_classes == 2:
        label = (label_img > 0).astype(np.uint8)
        pred = (pred_img > 0).astype(np.uint8)
        task = "binary"
    else:
        label = (label_img // 100).astype(np.uint8)
        pred = (pred_img // 100).astype(np.uint8)
        task = "multiclass"

    # ---- 转 tensor，加 batch 维度 (B,H,W) ----
    label = torch.from_numpy(label).long().unsqueeze(0).to(device)
    pred_mask = torch.from_numpy(pred).long().unsqueeze(0).to(device)

    print(f"label shape:     {label.shape}")
    print(f"pred_mask shape: {pred_mask.shape}")
    print(f"pred_mask 取值:  {pred_mask.unique().tolist()}")
    print(f"label 取值:      {label.unique().tolist()}")

    # ---- HD95 ----
    hd95 = HausdorffDistance(
        num_classes=num_classes,
        input_format='index',
        include_background=False,
        distance_metric='euclidean'
    ).to(device)
    hd95.update(pred_mask, label)
    print(f"HD95: {hd95.compute().item():.4f}")

    # ---- torchmetrics Accuracy ----
    if num_classes == 2:
        acc = torchmetrics.Accuracy(task='binary').to(device)
        prec = torchmetrics.Precision(task='binary').to(device)
        recall = torchmetrics.Recall(task='binary').to(device)
