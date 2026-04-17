# 🔥 Change Detection Laboratory (CDLab) 

> A unified and extensible PyTorch framework for **remote sensing change detection**, supporting multiple architectures, flexible configurations, and reproducible experiments.

## ⚡ Quick Start

Before training or inference, please first check the detailed instructions:

📚 **Documentation**

* 📂 Dataset configuration → [guide1](./docx/data_config_instruction.md)
* 🧩 Model registration → [guide2](./docx/new_model_register.md)
* ⚙️ Model configuration → [guide3](./docx/model_config_instruction.md)


---

## 🚀 Training

Once the configuration is ready, start training with:

```bash
python train.py --config ./config/STNet.yaml --gpu_ids 0
```

---

## 📌 Output Structure

All experiment outputs are automatically organized under:

```bash
./logs/{experiment_name}/
```

Example:

```bash
./logs/2026_04_02_09_32_57/
├── STNet.yaml                # ✅ merged full config (reproducibility, includes: dataset settings、model parameters、training hyperparameters)
├── weights/                  # model checkpoints
│   ├── best_metric_model.pth # ⭐ best model (recommended)
│   └── last.pth              # last checkpoint
├── loss/                     # training loss
│   ├── epoch_loss.txt        # loss per epoch
│   └── epoch_loss.png        # loss curve visualization
└── out.log                   # training logs (includes progress, metrics, and warnings)
```

During training, CDLab automatically saves a **fully merged configuration file**:

```bash
./logs/{timestamp}/STNet.yaml
```

👉 This file already contains:

* dataset configuration (`data_config`)
* model parameters
* training settings

---

## 🔍 Inference (Prediction)

Run inference using:

```bash
python predict.py \
  --test_config ./logs/xxx/STNet.yaml \
  --weight_path ./logs/xxx/weights/best_metric_model.pth \
  --output_dir ./work_dirs
```

You **DO NOT need to reconfigure anything during inference**.

Simply provide: config + weight. Everything else is restored automatically.

Always use best_metric_model.pth instead of last.pth for inference.

---

## 🚀 Workflow Summary

```text
1. Read docs → understand config
2. Modify config/STNet.yaml
3. Train model
4. Use logs/xxx/STNet.yaml + best_metric_model.pth
5. Run predict.py
```

---

## 🎯 Design Philosophy

CDLab follows a **self-contained experiment design**:

* Each experiment folder = **fully reproducible unit**
* No need to reconfigure dataset or model
* Training and inference are fully decoupled

---
