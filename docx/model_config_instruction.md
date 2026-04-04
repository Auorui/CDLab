### 🧪 Model Configuration

Each experiment in CDL ab is driven by a YAML configuration file located in:

```bash id="2g0g4c"
./config/
```

---

### 🔹 Example: `config/STNet.yaml`

```yaml id="9v1qdn"
data_config: '__base__/CLCD.yaml'

model:
  name: 'STNet'
  params:
    backbone: 'resnet18'
    num_classes: 2
    channel_list: [64, 128, 256, 512]
    transform_feat: 128
    layer_num: 4
    pretrained: true

optimizer_type: 'adamw'
lr: 0.0001
momentum: 0.9
weight_decay: 0.0001

scheduler_type: 'gradual_warm'
warmup_epochs: 5

loss_type: ['dice', 'focal']
loss_weight: [0.5, 0.5]
aux_loss_weights: null

epochs: 200
batch_size: 2

log_dir: './logs'
save_period: 50
resume_training: null
```

---

### ⚡ Minimal Setup to Start Training

To launch training, you only need to modify **two key fields**:

---

#### 1️⃣ Dataset Configuration

```yaml id="t1r6qk"
data_config: '__base__/CLCD.yaml'
```

👉 Replace with your dataset:

```yaml id="6p4pzz"
data_config: '__base__/YourDataset.yaml'
```

---

#### 2️⃣ Number of Classes

```yaml id="kbtq7m"
model:
  params:
    num_classes: 2
```

👉 Must be consistent with:

```yaml id="qxy8v4"
data.num_classes
```

Inconsistency will raise an error during config loading.

---

### ⚠️ Important Notes

* For **binary change detection**:

  * `num_classes = 2`
  * Label mapping must be:

    ```
    0 → NotChanged
    1 → Changed
    ```

* For **multi-class change detection**:

  * Set `num_classes > 2`
  * Ensure `color_map` is correctly defined

---

### ⚙️ Optimizer

```yaml id="5q1r6w"
optimizer_type: 'adamw'
lr: 0.0001
momentum: 0.9
weight_decay: 0.0001
```

📌 Supported optimizers and parameters are defined in:

```bash id="cz92i3"
pyzjr.nn.optim.get_optimizer
```

---

### 📉 Learning Rate Scheduler

```yaml id="n7w7hb"
scheduler_type: 'gradual_warm'
warmup_epochs: 5
```

📌 Available schedulers are implemented in:

```bash id="u4fskz"
pyzjr.nn.optim.get_lr_scheduler
```

---

### 📊 Loss Function

```yaml id="9r5r4f"
loss_type: ['dice', 'focal']
loss_weight: [0.5, 0.5]
```

📌 Supported loss functions are defined in:

```bash id="b8o1ql"
./utils/losses.py
```

Available options include:

* `ce` (Cross Entropy)
* `dice`
* `focal`
* `iou`

---

### 🔧 Training Control

```yaml id="qv7z0f"
epochs: 200
batch_size: 2
save_period: 50
```

---

### 🔁 Resume Training

```yaml id="w2lm6u"
resume_training: './logs/STNet/best_model.pth'
```

👉 Set to `null` to train from scratch

---

### 🚀 Run Training

```bash id="km6p2h"
python train.py --config ./config/STNet.yaml --gpu_ids 0
```

---

### 💡 Summary

To start a new experiment:

1. Select dataset via `data_config`
2. Set correct `num_classes`
3. Choose model (`model.name`)
4. Configure optimizer / scheduler / loss (optional)
5. Run training script

---
