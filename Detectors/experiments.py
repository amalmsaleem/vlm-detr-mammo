import collections
import cv2
import gc
import os
import pandas as pd
import pickle
import random
import time
import torch
import warnings
from pathlib import Path
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from Datasets.dataset_utils import get_dataset, get_transforms
from Detectors.detectors_utils import _get_detections, _get_annotations, compute_overlap, _compute_ap
from Detectors.retinanet.detector_model import RetinaNet_efficientnet
from utils import seed_all, timeSince

#detr
from Detectors.retinanet.engine import evaluate, train_one_epoch
import Detectors.retinanet.misc as utils
from Detectors.retinanet.coco_eval import CocoEvaluator
from Detectors.retinanet.detr import DETR_efficientnet
import json
import datetime

warnings.filterwarnings("ignore")
import numpy as np

selected_image_indices = None 



def do_experiements(args, device):
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    print(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    print(f"df shape: {args.df.shape}")

    if args.dataset.lower() == 'vindr':
        args.df = args.df.head(2254)
    args.model_base_name = args.arch
    seed_all(args.seed)
    args.cur_fold = 0
    args.train_folds = args.df[args.df['split'] == "training"].reset_index(drop=True)
    args.valid_folds = args.df[args.df['split'] == "test"].reset_index(drop=True)
    print(f'train: {args.train_folds.shape}', f'valid: {args.valid_folds.shape}')
    print(args.valid_folds.columns)
    if args.running_interactive:
        # test on small subsets of data on interactive mode
        args.train_folds = args.train_folds.head(100)
        args.valid_folds = args.valid_folds.head(n=1000)
        # print(args.valid_folds[args.valid_folds["Architectural_Distortion"] == 1].shape)

    train_loop(args, device)

    print(f'Checkpoints saved at: {args.chk_pt_path}')
    print(f'Outputs saved at: {args.output_path}')

import torchvision
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
#detr
def train_loop(args, device):
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    # if args.frozen_weights is not None:
    #     assert args.masks, "Frozen training is meant for segmentation only"
    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = 42 #+ utils.get_rank() #amal
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessors = DETR_efficientnet(args, device)
    model, criterion, postprocessors = DETR_efficientnet(num_classes=len(args.concepts), model_type=args.arch, 
                                                         clip_chk_pt=args.clip_chk_pt_path, freeze_backbone=args.freeze_backbone, device=device
    )
    model.to(device)

    model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200)

    data_loader_train, data_loader_val, valid_dataset = get_dataset(args)
    print(f'train_loader: {len(data_loader_train)}', f'valid_loader: {len(data_loader_val)}')

    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)

    # if args.distributed:
    #     sampler_train = DistributedSampler(dataset_train)
    #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # if args.dataset_file == "coco_panoptic":
    #     # We also evaluate AP during panoptic training, on original coco DS
    #     coco_val = datasets.coco.build("val", args)
    #     base_ds = get_coco_api_from_dataset(coco_val)
    # else:
    #     base_ds = get_coco_api_from_dataset(dataset_val)

    base_ds = get_coco_api_from_dataset(valid_dataset)

    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_path)
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         args.start_epoch = checkpoint['epoch'] + 1

    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_path)
    #     if args.output_path:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            0.1)
        lr_scheduler.step()
        if args.output_path:
            checkpoint_paths = [args.chk_pt_path / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(args.output_path / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_path
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_path and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
