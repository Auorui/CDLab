# 🔥 Change Detection Laboratory (CDLab) 

> A unified and extensible PyTorch framework for **remote sensing change detection**, supporting multiple architectures, flexible configurations, and reproducible experiments.

## ⚡ Quick Start

Before training or inference, please first check the detailed instructions:

📚 **Documentation**

* 📂 Dataset configuration → [data_config_instruction.md](./docx/data_config_instruction.md)
* 🧩 Model registration → [new_model_register.md](./docx/new_model_register.md)
* ⚙️ Model configuration → [model_config_instruction.md](./docx/model_config_instruction.md)


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
├── STNet.yaml                # ✅ merged full config (reproducibility)
├── weights/                  # model checkpoints
│   ├── best_metric_model.pth # ⭐ best model (recommended)
│   └── last.pth              # last checkpoint
├── loss/                     # training loss
│   ├── epoch_loss.txt
│   └── epoch_loss.png
└── out.log                   # training logs
```

---

## 🔍 Output Explanation

### 📄 `STNet.yaml`

* Fully merged configuration file
* Includes:

  * dataset settings
  * model parameters
  * training hyperparameters
    👉 Used directly for **inference and reproducibility**

---

### 🧠 `weights/`

* `best_metric_model.pth` → best model based on evaluation metric (**recommended**)
* `last.pth` → final checkpoint

---

### 📉 `loss/`

* `epoch_loss.txt` → loss per epoch
* `epoch_loss.png` → loss curve visualization

---

### 📝 `out.log`

* Full training logs
* Includes progress, metrics, and warnings

---

## 🔍 Inference (Prediction)

Run inference using:

```bash
python predict.py \
  --test_config ./logs/xxx/STNet.yaml \
  --weight_path ./logs/xxx/weights/best_metric_model.pth \
  --output_dir ./work_dirs
```

---

## ⚠️ Important Design

During training, CDL ab automatically saves a **fully merged configuration file**:

```bash
./logs/{experiment}/STNet.yaml
```

👉 This file already contains:

* dataset configuration (`data_config`)
* model parameters
* training settings

---

## ✅ Why This Matters

You **DO NOT need to reconfigure anything during inference**.

Simply provide:

```bash
config + weight
```

Everything else is restored automatically.

---

## 💡 Recommended Practice

Always use:

```bash
best_metric_model.pth
```

instead of `last.pth` for inference.

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

## ✨ Key Advantages

* 🔥 No configuration mismatch during inference
* 🔥 Fully reproducible experiments
* 🔥 One-command prediction
* 🔥 Clean and structured logging system
* 🔥 Easy debugging and visualization

---
