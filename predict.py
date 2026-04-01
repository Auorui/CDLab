from typing import Any

import torch
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import pyzjr
import torchmetrics
from torch.utils.data import DataLoader

from models import get_change_networks
from utils import build_dataset, load_config, crop_scd_map


class CDInference(object):
    def __init__(self, args, device='cuda:0', threshold=.5):
        super(CDInference, self).__init__()
        self.device = torch.device(device)
        self.threshold = threshold
        self.only_index = args.only_index

        # 加载配置
        self.config = load_config(args.test_config, merge_data_config=False)

        # 创建模型
        model = get_change_networks(self.config.model.name, **self.config.model.params)
        self.model = model.to(self.device)

        # 加载权重
        checkpoint = torch.load(args.weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 数据配置
        self.data_config = self.config.data
        self.num_classes = self.data_config.num_classes
        self.target_shape = self.data_config.target_shape if hasattr(self.data_config, 'target_shape') else (512, 512)
        self.color_map = self.data_config.color_map if hasattr(self.data_config, 'color_map') else None

        if args.dataset_path:
            self.data_config.dataset_path = args.dataset_path

        # 保存位置
        self.output_dir = os.path.join(args.output_dir, self.config.model.name, pyzjr.timestr())
        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        self.binary_dir = os.path.join(self.output_dir, 'binary')
        pyzjr.multi_makedirs(self.vis_dir, self.binary_dir)

        print(f"Model loading completed, number of categories: {self.num_classes}")
        print(f"Target shape: {self.target_shape}")

        # 初始化 torchmetrics
        self._init_metrics()

    def _init_metrics(self):
        if self.num_classes == 2:
            # 二分类：建议用 binary 口径
            self.metric_acc = torchmetrics.Accuracy(task='binary').to(self.device)
            self.metric_prec = torchmetrics.Precision(task='binary').to(self.device)
            self.metric_recall = torchmetrics.Recall(task='binary').to(self.device)
            self.metric_f1 = torchmetrics.F1Score(task='binary').to(self.device)
            self.metric_iou = torchmetrics.JaccardIndex(task='binary').to(self.device)
        else:
            # 多分类
            metric_cfg = dict(
                task='multiclass',
                num_classes=self.num_classes,
                average='macro'
            )
            self.metric_acc = torchmetrics.Accuracy(**metric_cfg).to(self.device)
            self.metric_prec = torchmetrics.Precision(**metric_cfg).to(self.device)
            self.metric_recall = torchmetrics.Recall(**metric_cfg).to(self.device)
            self.metric_f1 = torchmetrics.F1Score(**metric_cfg).to(self.device)
            self.metric_iou = torchmetrics.JaccardIndex(**metric_cfg).to(self.device)

    def reset_metrics(self):
        self.metric_acc.reset()
        self.metric_prec.reset()
        self.metric_recall.reset()
        self.metric_f1.reset()
        self.metric_iou.reset()

    def update_metrics(self, pred, label):
        """
        pred:
            binary: (B,H,W), 值为0/1
            multiclass: (B,H,W), 值为类别索引
        label:
            (B,H,W)
        """
        self.metric_acc.update(pred, label)
        self.metric_prec.update(pred, label)
        self.metric_recall.update(pred, label)
        self.metric_f1.update(pred, label)
        self.metric_iou.update(pred, label)

    def compute_metrics(self):
        results = {
            'Accuracy': self.metric_acc.compute().item(),
            'Precision': self.metric_prec.compute().item(),
            'Recall': self.metric_recall.compute().item(),
            'F1Score': self.metric_f1.compute().item(),
            'IoU': self.metric_iou.compute().item(),
        }
        return results

    def predict(self, mode='test', draw_color_map=None):
        """
        预测整个数据集, mode: 数据集模式（train/val/test）
        """
        dataset = build_dataset(self.config, mode=mode)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)

        print(f"Prediction dataset: {os.path.join(self.data_config.dataset_path, mode)}, containing {len(dataset)} image pairs")

        self.reset_metrics()

        for i, (img1, img2, label) in enumerate(tqdm(dataloader, desc=f"inference {mode} set")):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            label_torch = label.to(self.device).long()   # (B,H,W)
            label_np = label.squeeze(0).cpu().numpy()    # (H,W)

            with torch.no_grad():
                out = self.model(img1, img2)
            out = out[0] if isinstance(out, (list, tuple)) else out

            img_name = dataset.image_name_list[i]

            if self.num_classes == 2:
                prob = torch.softmax(out, dim=1)[:, 1, :, :]   # (B,H,W)
                pred_torch = (prob > self.threshold).long()    # (B,H,W)
            else:
                pred_torch = torch.argmax(out, dim=1)          # (B,H,W)

            # 更新指标
            self.update_metrics(pred_torch, label_torch)

            # 保存结果
            if not self.only_index:
                self.save_prediction(pred_torch, label_np, img_name, draw_color_map)

        # 计算最终指标
        metrics = self.compute_metrics()
        print("\n===== TorchMetrics Results =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

        # 保存指标到文件
        metric_file = os.path.join(self.output_dir, "metric_results.txt")
        with open(metric_file, "w", encoding="utf-8") as f:
            f.write(f"Prediction dataset: {os.path.join(self.data_config.dataset_path, mode)}, containing {len(dataset)} image pairs")
            f.write("===== TorchMetrics Results =====\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")

        print(f"\nMetrics saved to: {metric_file}")

    def save_prediction(self, pred_torch, label=None, img_name=None, draw_color_map=None, confusion_map=True):
        pred = pred_torch.squeeze(0).cpu().numpy()

        if self.num_classes == 2:
            binary = (pred * 255).astype(np.uint8)
            binary_save_dir = os.path.join(self.binary_dir, img_name)
            cv2.imwrite(binary_save_dir, binary)

            if confusion_map and label is not None:
                vis_save_dir = os.path.join(self.vis_dir, img_name)
                vis = self.visualize_confusion(pred, label)
                cv2.imwrite(vis_save_dir, vis)
        else:
            if self.color_map is None and self.num_classes > 2:
                self.color_map = self._generate_color_palette(self.num_classes)

            color_pred = self.visualize_multiclass(pred, draw_color_map)
            vis_save_path = os.path.join(self.vis_dir, img_name)
            cv2.imwrite(vis_save_path, color_pred)

    def visualize_multiclass(self, mask, draw_color_map=None):
        colors_map: Any | None = self.color_map if draw_color_map is None else draw_color_map
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, color in colors_map.items():
            if class_idx < self.num_classes:
                color_mask[mask == class_idx] = color
        return color_mask

    def visualize_confusion(self, pred, gt):
        """
        TP: 白色
        TN: 黑色
        FP: 红色
        FN: 绿色
        """
        h, w = pred.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        tp = (pred == 1) & (gt == 1)
        vis[tp] = [255, 255, 255]

        fp = (pred == 1) & (gt == 0)
        vis[fp] = [0, 0, 255]

        fn = (pred == 0) & (gt == 1)
        vis[fn] = [0, 255, 0]

        return vis

    def _generate_color_palette(self, num_classes):
        if num_classes <= 9:
            color_dict = {
                0: np.array([255, 255, 255]),  # No Cropland Change
                1: np.array([0, 0, 255]),      # Water
                2: np.array([0, 100, 0]),      # Forest
                3: np.array([0, 128, 0]),      # Plantation
                4: np.array([0, 255, 0]),      # Grassland
                5: np.array([128, 0, 0]),      # Impervious surface
                6: np.array([0, 255, 255]),    # Road
                7: np.array([255, 0, 0]),      # Greenhouse
                8: np.array([255, 192, 0]),    # Bare soil
            }
            return color_dict
        else:
            raise NotImplementedError("Please define color palette for num_classes > 9")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change Detection with YAML Config Test')
    parser.add_argument('--test_config', type=str, default=r'E:\PythonProject\CDLab\weights\CropSCD\MeGNet\MeGNet.yaml',
                        help='path to config file')
    parser.add_argument('--weight_path', default=r'E:\PythonProject\CDLab\weights\CropSCD\MeGNet\best_metric_model.pth', type=str,
                        help='path to models saving')
    parser.add_argument("--output_dir", type=str, default=r'./work_dirs')
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--only_index", type=bool, default=True)
    args = parser.parse_args()

    cd_infer = CDInference(args, device=args.device, threshold=args.threshold)
    cd_infer.predict('test', draw_color_map=None)