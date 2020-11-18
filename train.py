import time
import logging
import os, sys
import argparse
from collections import deque
import datetime

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from easydict import EasyDict as edict

from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4

from loss import Yolo_loss
from utils.utils import collate_fn as val_collate
from utils.utils import collate


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5, freeze_backbone=False):
    """ Train the YOLOv4 network with given configurations """
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True, collate_fn=val_collate)

    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:      {config.pretrained}
    ''')

    # Learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # Optimiser alternatives
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    # Set model to training mode
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_step = 0

        # Freeze backbone first epoch
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if not 'head' in name.split('.')[0]:  # if layer < 137
                    p.requires_grad = False if (epoch == 0) else True

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=200) as progress_bar:
            for batch in train_loader:
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    progress_bar.set_postfix(**{
                        'loss (batch)': loss.item(), 
                        'loss_xy': loss_xy.item(),
                        'loss_wh': loss_wh.item(),
                        'loss_obj': loss_obj.item(),
                        'loss_cls': loss_cls.item(),
                        'loss_l2': loss_l2.item(),
                        'lr': scheduler.get_lr()[0] * config.batch
                    })
                    logging.debug('Train step_{}: loss : {}, loss xy : {}, loss wh : {},'
                                  ' loss obj : {}, loss cls : {}, loss l2 : {}, lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_lr()[0] * config.batch))

                progress_bar.update(images.shape[0])

            eval_model = Yolov4(cfg.pretrained, n_classes=cfg.classes, inference=True)
            if torch.cuda.device_count() > 1:
                eval_model.load_state_dict(model.module.state_dict())
            else:
                eval_model.load_state_dict(model.state_dict())
            eval_model.to(device)

            """
            evaluator = evaluate(eval_model, val_loader, config, device)
            del eval_model

            stats = evaluator.coco_eval['bbox'].stats
            writer.add_scalar('train/AP', stats[0], global_step)
            writer.add_scalar('train/AP50', stats[1], global_step)
            writer.add_scalar('train/AP75', stats[2], global_step)
            writer.add_scalar('train/AP_small', stats[3], global_step)
            writer.add_scalar('train/AP_medium', stats[4], global_step)
            writer.add_scalar('train/AP_large', stats[5], global_step)
            writer.add_scalar('train/AR1', stats[6], global_step)
            writer.add_scalar('train/AR10', stats[7], global_step)
            writer.add_scalar('train/AR100', stats[8], global_step)
            writer.add_scalar('train/AR_small', stats[9], global_step)
            writer.add_scalar('train/AR_medium', stats[10], global_step)
            writer.add_scalar('train/AR_large', stats[11], global_step)
            """
            # Save model to file
            if save_cp:
                try:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'Checkpoint {epoch + 1} saved !')
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f'failed to remove {model_to_remove}')


def get_args(**kwargs):
    """ Append command line arguments to config dict """
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    args = vars(parser.parse_args())

    cfg.update(args)

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """ Logg log_level events to file """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    # Log destination
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)

    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Yolov4(cfg.pretrained, n_classes=cfg.classes)
    freeze_backbone = cfg.pretrained is not None

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device,
              freeze_backbone=freeze_backbone)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
