# Change Detection Laboratory (CDLab)

An experimental framework for remote sensing change detection based on PyTorch, supporting multiple network architectures and training configurations.

## 快速开始
1. 准备配置文件 (参考 config/STNet.yaml )
2. 运行训练脚本:
python train.py --config ./config/STNet.yaml --gpu_ids 0

配置文件
config/STNet.yaml
data_config: '__base__/CLCD.yaml'

model:
  name: 'STNet'
  params:
    backbone: 'resnet18'           # resnet18, resnet50_os8, resnet50_os32
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

epochs: 200
batch_size: 2

log_dir: './logs'
save_period: 50
resume_training: null      # 这里是完整的模型检查点路径（包含记忆项）


