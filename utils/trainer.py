import torch
import torchmetrics
from tqdm import tqdm
from pyzjr.nn import release_gpu_memory, get_lr, AverageMeter, SegmentationIndex, calculate_seg_confusion_matrix


class CDTrainEpoch(object):
    def __init__(self,
                 model,
                 model_name,
                 total_epoch,
                 optimizer,
                 loss_function,
                 num_classes=None,
                 device=torch.device("cuda:0")):
        super(CDTrainEpoch, self).__init__()
        self.model_name = model_name
        self.device = device
        self.model = model.to(device)
        self.criterion = loss_function
        self.optimizer = optimizer
        self.total_epoch = total_epoch
        self.num_classes = num_classes
        release_gpu_memory()

    # training
    def train_one_epoch(self, train_loader, epoch):
        train_bar = tqdm(train_loader,
                         desc=f'Epoch {epoch}/{self.total_epoch} [Train]',
                         mininterval=0.3)
        self.model.train()
        train_losses = AverageMeter()

        for img1, img2, label in train_bar:
            img1 = img1.to(self.device, dtype=torch.float)
            img2 = img2.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
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

            # 处理多输出模型
            if isinstance(prob, (list, tuple)):
                loss = sum([self.criterion(o, label) for o in prob])
            else:
                loss = self.criterion(prob, label)

            loss.backward()
            self.optimizer.step()

            train_losses.update(loss.item())
            train_bar.set_postfix(
                loss=f'{train_losses.avg:.4f}',
                lr=f'{get_lr(self.optimizer):.6f}'
            )
        return train_losses.avg

    # eval
    def evaluate(self, val_loader, epoch):
        val_bar = tqdm(val_loader,
                       desc=f'Epoch {epoch}/{self.total_epoch} [Val]',
                       mininterval=0.3)
        self.model.eval()
        val_losses = AverageMeter()

        # 初始化 torchmetrics
        if self.num_classes == 2:
            metric_acc = torchmetrics.Accuracy(task='binary').to(self.device)
            metric_prec = torchmetrics.Precision(task='binary').to(self.device)
            metric_recall = torchmetrics.Recall(task='binary').to(self.device)
            metric_f1 = torchmetrics.F1Score(task='binary').to(self.device)
            metric_iou = torchmetrics.JaccardIndex(task='binary').to(self.device)
        else:
            metric_cfg = dict(
                task='multiclass',
                num_classes=self.num_classes,
                average='macro'
            )
            metric_acc = torchmetrics.Accuracy(**metric_cfg).to(self.device)
            metric_prec = torchmetrics.Precision(**metric_cfg).to(self.device)
            metric_recall = torchmetrics.Recall(**metric_cfg).to(self.device)
            metric_f1 = torchmetrics.F1Score(**metric_cfg).to(self.device)
            metric_iou = torchmetrics.JaccardIndex(**metric_cfg).to(self.device)

        for img1, img2, label in val_bar:
            img1 = img1.to(self.device, dtype=torch.float)
            img2 = img2.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)

            if label.dim() == 4:
                label = label.squeeze(1)

            with torch.no_grad():
                out = self.model(img1, img2)

            pred = out[0] if isinstance(out, (list, tuple)) else out

            loss = self.criterion(pred, label)
            val_losses.update(loss.item())

            if self.num_classes == 2:
                # 二分类：取变化类概率
                prob = torch.softmax(pred, dim=1)[:, 1, :, :]   # (B,H,W)
                pred_mask = (prob > 0.5).long()                 # (B,H,W)
            else:
                # 多分类
                pred_mask = torch.argmax(pred, dim=1)           # (B,H,W)

            # (B, 1, H, W)
            # prob = torch.sigmoid(pred).squeeze(1)
            # pred_mask = (prob > 0.5).long()

            metric_acc.update(pred_mask, label)
            metric_prec.update(pred_mask, label)
            metric_recall.update(pred_mask, label)
            metric_f1.update(pred_mask, label)
            metric_iou.update(pred_mask, label)

            current_f1 = metric_f1.compute().item()
            val_bar.set_postfix(loss=f'{val_losses.avg:.4f}', f1=f'{current_f1:.4f}')

        val_acc = metric_acc.compute().item()
        val_prec = metric_prec.compute().item()
        val_recall = metric_recall.compute().item()
        val_f1 = metric_f1.compute().item()
        val_iou = metric_iou.compute().item()

        print("Validation Metrics:")
        print(f"Accuracy : {val_acc:.4f}")
        print(f"Precision: {val_prec:.4f}")
        print(f"Recall   : {val_recall:.4f}")
        print(f"F1 Score : {val_f1:.4f}")
        print(f"IoU      : {val_iou:.4f}\n")

        return val_losses.avg, val_f1


# 使用SegmentationIndex（仅作报留）
class CDTrainEpochIndex(object):
    def __init__(self,
                 model,
                 model_name,
                 total_epoch,
                 optimizer,
                 loss_function,
                 num_classes=None,
                 device=torch.device("cuda:0")):
        super(CDTrainEpochIndex, self).__init__()
        self.model_name = model_name  # 用于处理特殊模型接口
        self.device = device
        self.model = model.to(device)
        self.criterion = loss_function
        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        self.num_classes = num_classes
        release_gpu_memory()

    # training
    def train_one_epoch(self, train_loader, epoch):
        train_bar = tqdm(train_loader,
                         desc=f'Epoch {epoch}/{self.total_epoch} [Train]',
                         mininterval=0.3)
        self.model.train()
        train_losses = AverageMeter()

        for img1, img2, label in train_bar:
            img1 = img1.to(self.device, dtype=torch.float)
            img2 = img2.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            out = self.model(img1, img2)

            # 处理 MeGNet 的特殊输出(后续可添加其他类型的处理)
            if self.model_name == 'MeGNet' and isinstance(out, tuple):
                prob = out[0]
                if len(out) > 1:
                    self.memory_items = out[1].detach().clone()
                    if hasattr(self.model, 'set_memory_items'):
                        self.model.set_memory_items(self.memory_items)
            else:
                prob = out

            # 处理多输出模型（如辅助损失）
            if isinstance(prob, (list, tuple)):
                loss = sum([self.criterion(o, label) for o in prob])
            else:
                loss = self.criterion(prob, label)

            loss.backward()
            self.optimizer.step()
            train_losses.update(loss.item())
            train_bar.set_postfix(
                loss=f'{train_losses.avg:.4f}',
                lr=f'{get_lr(self.optimizer):.6f}'
            )
        return train_losses.avg

    # eval
    def evaluate(self, val_loader, epoch):
        val_bar = tqdm(val_loader,
                       desc=f'Epoch {epoch}/{self.total_epoch} [Val]',
                       mininterval=0.3)
        self.model.eval()
        val_losses = AverageMeter()

        all_tp, all_fn, all_fp, all_tn = [], [], [], []

        for img1, img2, label in val_bar:
            img1 = img1.to(self.device, dtype=torch.float)
            img2 = img2.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)

            with torch.no_grad():
                out = self.model(img1, img2)
            pred = out[0] if isinstance(out, (list, tuple)) else out

            loss = self.criterion(pred, label)
            val_losses.update(loss.item())
            if self.num_classes == 2:
                # ===== binary =====
                prob = torch.softmax(pred, dim=1)[:, 1:2, :, :]  # (B,1,H,W)
                target = label.unsqueeze(1)  # (B,1,H,W)

                tp, fn, fp, tn = calculate_seg_confusion_matrix(
                    prob,
                    target,
                    mode='binary',
                    threshold=0.5
                )

            else:
                # ===== multiclass =====
                pred = torch.argmax(pred, dim=1)  # (B,H,W)
                tp, fn, fp, tn = calculate_seg_confusion_matrix(
                    pred,
                    label,
                    mode='multiclass',
                    num_classes=self.num_classes
                )
            all_tp.append(tp)
            all_fn.append(fn)
            all_fp.append(fp)
            all_tn.append(tn)

            current_f1 = SegmentationIndex(tp, fn, fp, tn, reduction='micro').f1_score.item()
            val_bar.set_postfix(loss=f'{val_losses.avg:.4f}', f1=f'{current_f1:.4f}')
        # 合并结果
        total_tp = torch.cat(all_tp, dim=0)
        total_fn = torch.cat(all_fn, dim=0)
        total_fp = torch.cat(all_fp, dim=0)
        total_tn = torch.cat(all_tn, dim=0)
        # 计算各类指标
        cd_m = SegmentationIndex(total_tp, total_fn, total_fp, total_tn, reduction='micro')
        cd_m.eval()
        return val_losses.avg, cd_m.f1_score.item()




# 通用类型（仅做留存）
class CDTrainEpochBase(object):
    def __init__(self,
                 model,
                 model_name,
                 total_epoch,
                 optimizer,
                 loss_function,
                 num_classes=None,
                 device=torch.device("cuda:0")):
        super(CDTrainEpochBase, self).__init__()
        self.model_name = model_name  # 用于处理特殊模型接口
        self.device = device
        self.model = model.to(device)
        self.criterion = loss_function
        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        self.num_classes = num_classes
        release_gpu_memory()

    # training
    def train_one_epoch(self, train_loader, epoch):
        train_bar = tqdm(train_loader,
                         desc=f'Epoch {epoch}/{self.total_epoch} [Train]',
                         mininterval=0.3)
        self.model.train()
        train_losses = AverageMeter()

        for img1, img2, label in train_bar:
            img1 = img1.to(self.device, dtype=torch.float)
            img2 = img2.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            out = self.model(img1, img2)
            if isinstance(out, (list, tuple)):
                loss = sum([self.criterion(o, label) for o in out])
            else:
                loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            train_losses.update(loss.item())
            train_bar.set_postfix(
                loss=f'{train_losses.avg:.4f}',
                lr=f'{get_lr(self.optimizer):.6f}'
            )
        return train_losses.avg

    # eval
    def evaluate(self, val_loader, epoch):
        val_bar = tqdm(val_loader,
                       desc=f'Epoch {epoch}/{self.total_epoch} [Val]',
                       mininterval=0.3)
        self.model.eval()
        val_losses = AverageMeter()

        all_tp, all_fn, all_fp, all_tn = [], [], [], []

        for img1, img2, label in val_bar:
            img1 = img1.to(self.device, dtype=torch.float)
            img2 = img2.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)

            with torch.no_grad():
                out = self.model(img1, img2)
            pred = out[0] if isinstance(out, (list, tuple)) else out

            loss = self.criterion(pred, label)
            val_losses.update(loss.item())
            if self.num_classes == 2:
                # ===== binary =====
                prob = torch.softmax(pred, dim=1)[:, 1:2, :, :]  # (B,1,H,W)
                target = label.unsqueeze(1)  # (B,1,H,W)

                tp, fn, fp, tn = calculate_seg_confusion_matrix(
                    prob,
                    target,
                    mode='binary',
                    threshold=0.5
                )

            else:
                # ===== multiclass =====
                pred = torch.argmax(pred, dim=1)  # (B,H,W)
                tp, fn, fp, tn = calculate_seg_confusion_matrix(
                    pred,
                    label,
                    mode='multiclass',
                    num_classes=self.num_classes
                )
            all_tp.append(tp)
            all_fn.append(fn)
            all_fp.append(fp)
            all_tn.append(tn)

            current_f1 = SegmentationIndex(tp, fn, fp, tn, reduction='micro').f1_score.item()
            val_bar.set_postfix(loss=f'{val_losses.avg:.4f}', f1=f'{current_f1:.4f}')
        # 合并结果
        total_tp = torch.cat(all_tp, dim=0)
        total_fn = torch.cat(all_fn, dim=0)
        total_fp = torch.cat(all_fp, dim=0)
        total_tn = torch.cat(all_tn, dim=0)
        # 计算各类指标
        cd_m = SegmentationIndex(total_tp, total_fn, total_fp, total_tn, reduction='micro')
        cd_m.eval()
        return val_losses.avg, cd_m.f1_score.item()
