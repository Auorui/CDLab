import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
from models.network import get_change_networks
from utils import build_dataset, CDTrainEpoch, load_config, save_merged_config, CombinedLoss
import pyzjr
from pyzjr.data import loss_weights_dirs, TrainDataloader, EvalDataloader
from pyzjr.nn import LossHistory, get_optimizer, get_lr_scheduler
from pyzjr.utils import redirect_console

def parse_args(known=False):
    parser = argparse.ArgumentParser(description='Change Detection with YAML Config Train')
    parser.add_argument('--config', type=str, default='./config/STNet.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=11,
                        help='Random seed number, For example: 11, 42, 3407, 114514, 256')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use (e.g., 0,1,2)')
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    config = load_config(args.config)
    pyzjr.SeedEvery(args.seed)
    loss_log_dir, save_model_dir, timelog_dir = loss_weights_dirs(os.path.join(config.log_dir, config.model.name))
    redirect_console(os.path.join(timelog_dir, 'out.log'))
    pyzjr.show_config(args=args, yaml_path=args.config)
    save_merged_config(config, timelog_dir)
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')

    # selection model
    model_cfg = config.model
    network = get_change_networks(model_cfg.name, **model_cfg.params)

    network = network.to(device)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        print(f"Using multi-card training: GPU {gpu_ids}")
        network = torch.nn.DataParallel(network, device_ids=gpu_ids)
    else:
        print(f"Training with a single GPU: {device}")
    if config.resume is None:
        print(f"Initial training {model_cfg.name}")

    # load data
    # binary & multiclass classification change detection
    train_dataset = build_dataset(config, mode='train')
    val_dataset = build_dataset(config, mode='val')
    train_loader = TrainDataloader(train_dataset, batch_size=config.batch_size)
    val_loader = EvalDataloader(val_dataset, batch_size=2, num_workers=1)

    loss_history = LossHistory(loss_log_dir)
    optimizer = get_optimizer(
        network, optimizer_type=config.optimizer_type, init_lr=config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay)
    lr_scheduler = get_lr_scheduler(
        optimizer, scheduler_type=config.scheduler_type, init_lr=config.lr,
        warmup_epochs=config.warmup_epochs, total_epochs=config.epochs)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = CombinedLoss(config.loss_type, config.loss_weight, aux_loss_weights=config.aux_loss_weights,
                             num_classes=config.data.num_classes).to(device)
    cd_trainer = CDTrainEpoch(
        network, model_cfg.name, config.epochs, optimizer, lr_scheduler, criterion,
        num_classes=config.data.num_classes, resume=config.resume, device=device,
        ignore_index=config.data.ignore_index
    )
    cd_trainer.fit(
        train_loader, val_loader, save_model_dir, loss_history=loss_history,
        save_period=config.save_period, monitor_metric=config.monitor_metric
    )
