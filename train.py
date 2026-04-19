import os
import torch.utils.data
import argparse
from models.network import get_change_networks
from utils import build_dataset, CDTrainEpoch, load_config, save_merged_config, CombinedLoss
import pyzjr
from pyzjr.data import loss_weights_dirs, TrainDataloader, EvalDataloader
from pyzjr.nn import LossHistory, get_optimizer, get_lr_scheduler
from pyzjr.visualize.printf import redirect_console

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
    loss_log_dir, save_model_dir, timelog_dir = loss_weights_dirs(config.log_dir)
    redirect_console(os.path.join(timelog_dir, 'out.log'))
    pyzjr.show_config(args=args, yaml_path=args.config)
    save_merged_config(config, timelog_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')
    print(f"Using the device: {device}")

    # selection model
    model_cfg = config.model
    network = get_change_networks(model_cfg.name, **model_cfg.params)

    network = network.to(device)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        print(f"Using multi-card training: GPU {gpu_ids}")
        network = torch.nn.DataParallel(network, device_ids=gpu_ids)
    else:
        print(f"Training with a single GPU: GPU {device}")
    if config.resume_training is not None:
        print(f"Weights {config.resume_training} loaded into {model_cfg.name}")
        if torch.cuda.is_available():
            checkpoint = torch.load(config.resume_training, map_location=device, weights_only=True)
        else:
            checkpoint = torch.load(config.resume_training, map_location='cpu', weights_only=True)
        target_model = network.module if isinstance(network, torch.nn.DataParallel) else network
        model_dict = target_model.state_dict()
        filtered, skipped = {}, []
        for k, v in checkpoint.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        model_dict.update(filtered)
        target_model.load_state_dict(model_dict)
        print(f"Loaded: {len(filtered)} / {len(checkpoint)} layers")
        if skipped:
            print(f"  Skipped (shape mismatch or not found): {skipped}")
    else:
        print(f"Initial training {model_cfg.name}")

    # load data
    # binary classification change detection
    train_dataset = build_dataset(config, mode='train')
    val_dataset = build_dataset(config, mode='val')
    train_loader = TrainDataloader(train_dataset, batch_size=config.batch_size)
    val_loader = EvalDataloader(val_dataset, batch_size=2, num_workers=1)

    loss_history = LossHistory(loss_log_dir)
    optimizer = get_optimizer(
        network, optimizer_type=config.optimizer_type, init_lr=config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay)
    lr_scheduler = get_lr_scheduler(
        optimizer, scheduler_type='gradual_warm', init_lr=config.lr,
        warmup_epochs=config.warmup_epochs, total_epochs=config.epochs)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = CombinedLoss(config.loss_type, config.loss_weight, aux_loss_weights=config.aux_loss_weights,
                             num_classes=config.data.num_classes).to(device)
    cd_trainer = CDTrainEpoch(
        network, model_cfg.name, config.epochs, optimizer, criterion, num_classes=config.data.num_classes
    )
    for epoch in range(config.epochs):
        epoch = epoch + 1
        train_loss = cd_trainer.train_one_epoch(train_loader, epoch)
        val_loss, f1_score = cd_trainer.evaluate(val_loader, epoch)
        loss_history.append_loss(epoch, train_loss, val_loss)

        lr_scheduler.step()

        print('Epoch:' + str(epoch) + '/' + str(config.epochs))
        print('Total Loss: %.5f || Val Loss: %.5f ' % (train_loss, val_loss))

        pyzjr.SaveModelPth(
            network,
            save_dir=save_model_dir,
            metric=f1_score,
            epoch=epoch,
            total_epochs=config.epochs,
            save_period=config.save_period
        )
