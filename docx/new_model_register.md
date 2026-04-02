### 🧩 Add a New Model

CDLab adopts a **centralized model registry mechanism**.
All models are defined under:

```bash
./models/baseline/
```

and are **registered in a unified way** via:

```bash
./models/network.py
```

---

### 📌 Step-by-Step Guide

#### 1. Define your model

Place your model implementation inside:

```bash
./models/baseline/your_model.py
```

Example:

```python
class YourModel(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        ...
    
    def forward(self, x1, x2):
        ...
```

---

#### 2. Import the model

In `./models/network.py`, all models are imported via:

```python
from .baseline import *
```

👉 This means **any model defined in `baseline/` can be directly registered**.

---

#### 3. Register the model

Add your model to the `MODEL_CLASSES` dictionary:

```python
MODEL_CLASSES = {
    "MSCANet"          : MSCANet,
    "MeGNet"           : MeGNetApt,
    "STNet"            : STNet,
    "FC_SiamUnet_diff" : SiamUnet_diff,
    "SNUNet"           : SNUNet_ECAM,
    "ChangeFormer"     : ChangeFormerV6,
    "DCSI_UNet"        : DCSI_UNet,

    # 👉 Add your model here
    "YourModel"        : YourModel
}
```

---

#### 4. Use the model in config

Make sure the **model name in config matches the registry key**:

```yaml
model:
  name: 'YourModel'
  params:
    num_classes: 2
```

⚠️ **Important**:

* `model.name` must be exactly the same as the key in `MODEL_CLASSES`
* `params` must match your model constructor arguments

---

### ⚙️ How it Works

CDLab uses the following factory function:

```python
def get_change_networks(name, **kwargs):
    if name in MODEL_CLASSES:
        return MODEL_CLASSES[name](**kwargs)
    else:
        raise ValueError(f"Model {name} not found in MODEL_CLASSES")
```

👉 This enables:

* Plug-and-play model switching
* Fully config-driven experiments
* No need to modify training code

---

### 💡 Best Practices

* ✅ Keep model name **consistent across config and registry**
* ✅ Use clear naming (e.g., `STNet`, `ChangeFormer`)
* ✅ Avoid modifying training pipeline when adding new models
* ✅ Ensure your model output shape is `(B, C, H, W)`

---

### 🚀 Summary

To add a new model, you only need to:

1. Implement it in `baseline/`
2. Register it in `MODEL_CLASSES`
3. Set its name in the config

👉 No other code changes required.

---
