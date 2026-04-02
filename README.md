# 🔥 Change Detection Laboratory (CDLab)

> A unified and extensible PyTorch framework for **remote sensing change detection**, supporting multiple architectures, flexible configurations, and reproducible experiments.

---

## 🚀 Highlights

* ✅ **Modular design**: decoupled model / dataset / loss / config
* ✅ **YAML-driven experiments**: easy to reproduce and extend
* ✅ **Multi-model support**: plug-and-play architecture registry
* ✅ **Binary & multi-class change detection**
* ✅ **Flexible loss combinations** (CE, Dice, Focal, IoU)
* ✅ **TorchMetrics-based evaluation**
* ✅ **Supports multi-GPU training**

---

## 📁 Project Structure

```
CDLab/
├── config/                # Experiment configurations (YAML)
│   ├── __base__/          # Base dataset configs
│   └── STNet.yaml
├── models/                # Model definitions
│   ├── baseline/
│   └── network.py         # Model registry
├── utils/
│   ├── losses.py
│   ├── dataset.py
│   ├── trainer.py
│   └── config.py
├── train.py               # Training entry
├── infer.py               # Inference script
└── README.md
```

---

## ⚡ Quick Start

### 1. Prepare config file

Example: `config/STNet.yaml`

```yaml
data_config: '__base__/CLCD.yaml'

model:
  name: 'STNet'
  params:
    backbone: 'resnet18'
    num_classes: 2
```

---

### 2. Train

```bash
python train.py --config ./config/STNet.yaml --gpu_ids 0
```

---

### 3. Inference

```bash
python infer.py \
  --test_config ./weights/STNet/STNet.yaml \
  --weight_path ./weights/STNet/best_model.pth \
  --output_dir ./results
```

---

## ⚙️ Configuration System

CDLab uses a **hierarchical YAML configuration system**.

### 🔹 Main Config

```yaml
model:
  name: 'STNet'
  params:
    backbone: 'resnet18'
    num_classes: 2

optimizer_type: 'adamw'
lr: 1e-4

loss_type: ['dice', 'focal']
loss_weight: [0.5, 0.5]
```

---

### 🔹 Dataset Config

```yaml
data:
  dataset_path: 'data/CLCD'
  target_shape: 512
  num_classes: 2
  dir_n1: 'image1'
  dir_n2: 'image2'

  color_map:
    NotChanged: [0, 0, 0]
    Changed: [255, 255, 255]
```

---

### ✅ Features

* Automatic **config inheritance**
* Consistency check:

  * `data.num_classes == model.num_classes`
* Automatic config saving for reproducibility

---

## 🧠 Supported Models

| Model        | Description              |
| ------------ | ------------------------ |
| STNet        | Spatio-temporal modeling |
| SNUNet       | Siamese nested U-Net     |
| ChangeFormer | Transformer-based CD     |
| DCSI-UNet    | CNN-based CD             |
| MeGNet       | Memory-guided network    |
| MSCANet      | Multi-scale attention    |

👉 Easily extend via:

```python
MODEL_CLASSES = {
    "YourModel": YourModel
}
```

---

## 📊 Loss Functions

CDLab supports flexible combinations:

* Cross Entropy
* Dice Loss
* Focal Loss
* IoU (Jaccard Loss)

Example:

```yaml
loss_type: ['dice', 'focal']
loss_weight: [0.5, 0.5]
```

---

## 📈 Evaluation Metrics

Based on **TorchMetrics**:

* Accuracy
* Precision
* Recall
* F1 Score
* IoU

---

## 📦 Dataset Format

### Binary Change Detection

```
dataset/
├── train/
│   ├── image1/
│   ├── image2/
│   └── label/
├── val/
└── test/
```

---

### Multi-class Change Detection

Supports semantic change detection with customizable `color_map`.

---

## 🧪 Training Pipeline

* Automatic logging
* Learning rate scheduling (warmup supported)
* Multi-output loss support
* Best model saving based on F1-score

---

## 🎯 Inference & Visualization

* Binary mask output
* Confusion map visualization:

  * TP: White
  * FP: Red
  * FN: Green

---

## 🔧 Customization

### Add a new model

```python
from models.network import MODEL_CLASSES

MODEL_CLASSES["NewModel"] = NewModel
```

---

### Add a new dataset

Implement:

```python
class YourDataset(BaseDataset):
    ...
```

---

## 📌 TODO

* [ ] Add more SOTA models
* [ ] Support distributed training (DDP)
* [ ] Add visualization dashboard
* [ ] Benchmark on public datasets

---

## 🤝 Acknowledgement

This project is built upon:

* PyTorch
* TorchMetrics
* Custom utility library `pyzjr`

---


---

## 📬 Contact

Feel free to open issues or contact for collaboration.

---
