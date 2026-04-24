### 📂 Dataset Configuration

All dataset configurations are defined under:

```bash
./config/__base__/
```

Each dataset is described using a YAML file, which is automatically loaded and merged into the main config.

We provide a brief introduction, download links, and the citation paper for a remote sensing change detection dataset via a [blog post](https://blog.csdn.net/m0_62919535/article/details/158573055)

---

### 🔹 Example: Binary Change Detection (CLCD)

```yaml
data:
  dataset_path: 'data/CLCD'
  target_shape: 512
  num_classes: 2

  # directory names for t1 and t2 images
  dir_n1: 'image1'
  dir_n2: 'image2'

  # label mapping (RGB → class index)
  color_map:
    NotChanged: [0, 0, 0]        # label = 0
    Changed: [255, 255, 255]     # label = 1  ⚠️ MUST be 1
```

---

### 📁 Dataset Structure

```bash
dataset/
├── train/
│   ├── image1/     # t1 images
│   ├── image2/     # t2 images
│   └── label/      # RGB label maps
├── val/
└── test/
```

---

### ⚠️ Important Note (Critical)

For **binary change detection**, the label mapping must follow:

```text
0 → NotChanged (background)
1 → Changed (foreground)
```

❗ This is **strictly required** because:

* The Dice Loss is computed **only on the foreground class**
* The implementation assumes:

```python
prob_fg = prob[:, 1, :, :]
```

👉 If your dataset uses a different convention (e.g., Changed = 0),
you **must remap the labels** before training.

---

### 🔹 Multi-class Change Detection (CropSCD)

```yaml
data:
  dataset_path: 'data/CropSCD'
  target_shape: 512
  num_classes: 9

  dir_n1: 'im1'
  dir_n2: 'im2'

  color_map: null
#            {   The dataset has already been processed, so no mapping is required.
#      0: [255, 255, 255],       # No Cropland Change   
#      1: [0, 0, 255],           # Water
#      2: [0, 100, 0],           # Forest
#      3: [0, 128, 0],           # Plantation
#      4: [0, 255, 0],           # Grassland
#      5: [128, 0, 0],           # Impervious surface
#      6: [0, 255, 255],         # Road
#      7: [255, 0, 0],           # Greenhouse
#      8: [255, 192, 0],         # Bare soil
#}
```

---

### ⚙️ How It Works

The dataset builder automatically selects the dataset type:

```python
if num_classes == 2:
    CLCDataset
else:
    CropSCDataset
```

* Binary tasks → `CLCDataset`
* Multi-class tasks → `CropSCDataset`

---

### 💡 Tips

* ✅ Ensure **image1 / image2 / label filenames are identical**
* ✅ Labels must be **RGB images**, not index maps (unless color_map=None)
* ✅ Input images are automatically normalized to **[-1, 1]**
* ✅ Data augmentation is applied only during training

---

### 🚀 Summary

To use a new dataset:

1. Define a YAML file in `config/__base__/`
2. Set correct `color_map` (especially for binary tasks)
3. Ensure folder structure is consistent
4. Reference it via:

```yaml
data_config: '__base__/your_dataset.yaml'
```

---
