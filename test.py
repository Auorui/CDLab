import os
import cv2
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

import pyzjr
from pyzjr.nn import load_model_weights
from pyzjr.data import logits2color, error2color
from pyzjr.aug.geometric import resize_pad, resize_unpad

from models import get_change_networks
from utils import load_config, extract_logits, build_dataset, CDMetrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation with YAML Config Test')
    parser.add_argument('--test_config', type=str,
                        default=r'E:\PythonProject\CDLab\weights\LEVIR_CD\DCSI_UNet\DCSI_UNet.yaml',
                        help='path to config file')
    parser.add_argument('--weight_path',
                        default=r'E:\PythonProject\CDLab\weights\LEVIR_CD\DCSI_UNet\best_metric_model.pth', type=str,
                        help='path to models saving')
    parser.add_argument("--output_dir", type=str, default=r'./work_dirs')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--error_map", type=bool, default=True,
                        help="Save TP/TN/FP/FN error map instead of class color map")
    parser.add_argument("--only_index", type=bool, default=False, help="Only save single channel index mask")
    args = parser.parse_args()

    config = load_config(args.test_config, merge_data_config=False)

    if args.error_map and config.data.num_classes > 2:
        print(
            f"[Warning] error_map is only supported for binary segmentation "
            f"(num_classes == 2), but num_classes = {config.data.num_classes}. "
            f"Disabling error_map and falling back to class color map."
        )
        args.error_map = False

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(args.output_dir, config.model.name, pyzjr.timestr())
    if not args.only_index:
        os.makedirs(output_dir, exist_ok=True)

    model = get_change_networks(config.model.name, **config.model.params)
    model = model.to(device)
    print(f"Loading weights from: {args.weight_path}")
    model = load_model_weights(model, args.weight_path, map_location=device)
    model.eval()

    test_dataset = build_dataset(config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)

    metrics = CDMetrics(config.data.num_classes, device=device)

    with torch.no_grad():
        for i, (img1, img2, label, pad_meta, image_name) in enumerate(test_loader):
            img1 = img1.to(device, dtype=torch.float32)
            img2 = img2.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)

            logits = extract_logits(model(img1, img2))
            pred = logits.argmax(dim=1)                       # (B,H,W)
            metrics.update(pred, label)
            pred_mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)   # (H,W)
            gt_mask = label.squeeze(0).cpu().numpy().astype(np.uint8)
            if not all(int(x.item()) == 0 for x in pad_meta[:2]):
                pred_mask = resize_unpad(pred_mask, pad_meta, is_label=True)
                gt_mask = resize_unpad(gt_mask, pad_meta, is_label=True)

            image_name = image_name[0] if isinstance(image_name, (list, tuple)) else image_name
            name = os.path.splitext(os.path.basename(image_name))[0]
            if args.error_map:
                vis = error2color(pred_mask, gt_mask,
                                  ignore_index=256,
                                  rgb_output=False)
                save_path = os.path.join(output_dir, f"{name}_error.png")
            else:
                vis = logits2color(pred_mask,
                                  num_classes=config.data.num_classes,
                                  color_map=config.data.color_map,
                                  rgb_output=False)
                save_path = os.path.join(output_dir, f"{name}.png")

            if not args.only_index:
                ok = cv2.imwrite(save_path, vis)
                if not ok:
                    print(f"imwrite FAILED: {save_path}")

            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{len(test_loader)}] processed")

    results = metrics.compute()
    print("\n===== Test Results =====")
    pyzjr.printc(results)
