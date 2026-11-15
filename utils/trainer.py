import torch
from tqdm import tqdm
from utils.metrics import CDMetrics, print_metrics_table
from pyzjr.nn import release_gpu_memory, get_lr, AverageMeter, load_checkpoint, save_checkpoint


def extract_logits(output):
    """Normalize common model output conventions to a logits tensor."""
    if isinstance(output, dict):
        for key in ("out", "logits", "main"):
            if key in output:
                return extract_logits(output[key])
        return extract_logits(next(iter(output.values())))
    if isinstance(output, (list, tuple)):
        return extract_logits(output[0])
    if not torch.is_tensor(output):
        raise TypeError(f"Model output must be a tensor, got {type(output)!r}")
    return output


class CDTrainEpoch:
    def __init__(
            self,
            model,
            model_name,
            total_epoch,
            optimizer,
            lr_scheduler,
            loss_function,
            num_classes,
            resume=None,
            device=None,
            grad_clip=None,
            ignore_index=None
    ):
        self.device = device
        self.model = model.to(device)
        self.model_name = model_name
        self.criterion = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.resume = resume
        self.total_epoch = total_epoch
        self.num_classes = num_classes
        self.grad_clip = grad_clip
        self.ignore_index = ignore_index
        self.start_epoch = 1
        self.best_metric = float('-inf')
        release_gpu_memory()

    @staticmethod
    def _move_batch(batch, device):
        img1, img2, label = batch
        img1 = img1.to(device, dtype=torch.float32, non_blocking=True)
        img2 = img2.to(device, dtype=torch.float32, non_blocking=True)
        label = label.to(device, dtype=torch.long, non_blocking=True)
        return img1, img2, label

    def load_resume_ckpt(self, checkpoint_path, strict=True):
        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            self.model.load_state_dict(ckpt['state_dict'], strict=strict)
            if ckpt.get('optimizer') is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.lr_scheduler is not None and ckpt.get('lr_scheduler') is not None:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch = int(ckpt.get('epoch', 0)) + 1
            self.best_metric = ckpt.get('best_metric', float('-inf'))
        else:
            state_dict = ckpt.state_dict() if isinstance(ckpt, torch.nn.Module) else ckpt
            self.model.load_state_dict(state_dict, strict=strict)
            self.start_epoch = 1
            self.best_metric = float('-inf')

        print(f"Resuming from epoch {self.start_epoch}, Best Metric: {self.best_metric:.4f}")

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        train_losses = AverageMeter()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.total_epoch} [Train]",
                         leave=False, mininterval=0.3)
        for batch in train_bar:
            img1, img2, label = self._move_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            out = self.model(img1, img2)

            # 处理 MeGNet 的特殊输出
            if self.model_name == 'MeGNet' and isinstance(out, tuple):
                prob = out[0]
                if len(out) > 1:
                    self.memory_items = out[1].detach().clone()
                    if hasattr(self.model, 'set_memory_items'):
                        self.model.set_memory_items(self.memory_items)
            else:
                prob = out

            loss = self.criterion(prob, label)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            train_losses.update(loss.item(), n=img1.size(0))
            train_bar.set_postfix(
                loss=f'{train_losses.avg:.4f}',
                lr=f'{get_lr(self.optimizer):.6f}'
            )
        return train_losses.avg

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        val_losses = AverageMeter()
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{self.total_epoch} [Val]',
                       leave=False, mininterval=0.3)
        metrics = CDMetrics(self.num_classes, device=self.device, ignore_index=self.ignore_index)
        for batch in val_bar:
            img1, img2, label = self._move_batch(batch, self.device)
            if label.dim() == 4:
                label = label.squeeze(1)
            with torch.no_grad():
                out = self.model(img1, img2)

            logits = extract_logits(out)
            loss = self.criterion(logits, label)

            prediction = logits.argmax(dim=1)
            # (B, 1, H, W)
            # prob = torch.sigmoid(logits).squeeze(1)
            # prediction = (prob > 0.5).long()

            metrics.update(prediction, label)
            val_losses.update(loss.item(), n=img1.size(0))
            val_bar.set_postfix(loss=f'{val_losses.avg:.4f}')
        return val_losses.avg, metrics.compute()

    def fit(self, train_loader, val_loader, save_model_dir, loss_history=None,
            save_period=80, monitor_metric="IoU"):
        if self.resume:
            self.load_resume_ckpt(self.resume)
        if monitor_metric not in ['OA', 'P', 'R', 'F1', 'IoU']:
            raise KeyError(f"monitor_metric={monitor_metric!r} not in metrics: ['OA', 'P', 'R', 'F1', 'IoU']")

        for epoch in range(self.start_epoch, self.total_epoch + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, metrics_info = self.evaluate(val_loader, epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if loss_history is not None:
                loss_history.append_loss(epoch, train_loss, val_loss)

            print_metrics_table(metrics_info, epoch, train_loss, val_loss)

            current_metric = metrics_info.get(monitor_metric, 0.0)

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                save_checkpoint(
                    save_dir=save_model_dir,
                    filename="best_model",
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    val_loss=val_loss,
                    best_metric=self.best_metric,
                    metrics_info=metrics_info
                )

            if epoch % save_period == 0:
                save_checkpoint(
                    save_dir=save_model_dir,
                    filename=f"checkpoint_epoch_{epoch}",
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    val_loss=val_loss,
                    best_metric=self.best_metric,
                    metrics_info=metrics_info
                )
        if loss_history is not None:
            loss_history.finalize()

        # pyzjr.SaveModelPth(
        #     network,
        #     save_dir=save_model_dir,
        #     metric=f1_score,
        #     epoch=epoch,
        #     total_epochs=config.epochs,
        #     save_period=config.save_period
        # )