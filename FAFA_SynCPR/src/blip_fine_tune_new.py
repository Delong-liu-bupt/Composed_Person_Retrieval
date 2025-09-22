# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR
import os
import sys
from sampler import RandomIdentitySampler, RandomIdnumSampler, MultiCprBatchSampler
from data_utils import squarepad_transform, targetpad_transform, train_transform, inference_transform, SynCPRDataset, ITCPRDataset, GalleryDataset, QueryDataset, squarepad_transform_train, squarepad_transform_test
from utils import collate_fn, update_train_running_results,update_train_running_results_dict, set_train_bar_description_dict,set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip import compute_ticpr_val_metrics


def backup_code(source_dir, log_dir):
    import shutil
    code_backup_dir = os.path.join(log_dir, 'code_backup')
    if not os.path.exists(code_backup_dir):
        os.makedirs(code_backup_dir)
    common_prefix = os.path.commonprefix([source_dir, code_backup_dir])
    for root, dirs, files in os.walk(source_dir):
        if os.path.basename(root) in ['output', '__pycache__', 'submission']:
            dirs[:] = []
            continue
        for file in files:
            if file.endswith(('ipynb', 'md')):
                continue
            source_file = os.path.join(root, file)
            backup_file = os.path.join(code_backup_dir, os.path.relpath(source_file, common_prefix))
            shutil.copy2(source_file, backup_file)
        for dir in dirs:
            if dir in ['output', '__pycache__', 'submission']:
                continue
            source_dir = os.path.join(root, dir)
            backup_dir = os.path.join(code_backup_dir, os.path.relpath(source_dir, common_prefix))
            os.makedirs(backup_dir, exist_ok=True)



def clip_finetune_cpr(num_epochs: int, exp_name: str, blip_model_name: str, backbone: str, resume_path: str, learning_rate: float, batch_size: int, setting: List[str],
                       validation_frequency: int, validation_step: int, transform: str, save_training: bool, save_best: bool, save_last: bool, json_path: str,
                       syncpr_data_path: str, itcpr_root: str, **kwargs):
    """
    Fine-tune FAFA on the SynCPR dataset for Composed Person Retrieval (CPR) task
    :param num_epochs: number of epochs
    :param blip_model_name: BLIP model you want to use
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use
    :param save_training: when True save the weights of the model
    :param save_best: when True save only the weights of the best model
    :param kwargs: additional hyperparameters including FDA and FD settings
    """
    # rtc_weights = kwargs['loss_rtc']
    # align_weights = kwargs['loss_align']
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(f"./output/cpr/{training_start}_{exp_name}")
    training_path.mkdir(exist_ok=False, parents=True)
    backup_code('./', training_path)
    
    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)
    
    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, model_path=resume_path, is_eval=False, device=device)
    # print(blip_model)

    # Set FDA and FD parameters from command line arguments
    if hasattr(blip_model, 'fda_k'):
        blip_model.fda_k = kwargs.get('fda_k', 6)
        print(f"Setting FDA k={blip_model.fda_k} for top-k feature selection")
    if hasattr(blip_model, 'fda_alpha'):
        blip_model.fda_alpha = kwargs.get('fda_alpha', 0.5)
        print(f"Setting FDA alpha={blip_model.fda_alpha} for soft label strength")
    if hasattr(blip_model, 'dispersion_loss') and hasattr(blip_model.dispersion_loss, 'margin'):
        blip_model.dispersion_loss.margin = kwargs.get('fd_margin', 0.5)
        print(f"Setting FD margin={blip_model.dispersion_loss.margin}")

    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    # clip_model.eval().float()
    input_dim = 224

    if transform == "squarepad":
        # preprocess = squarepad_transform(input_dim)
        train_preprocess = squarepad_transform_train(input_dim)
        test_preprocess = squarepad_transform_test(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    elif transform == "resize":
        train_preprocess = train_transform(input_dim)
        test_preprocess = inference_transform(input_dim)
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # Define the validation datasets
    val_dataset = ITCPRDataset(root=itcpr_root)
    if transform == "resize" or transform == "squarepad":
        ds = val_dataset.query
        val_query_set = QueryDataset(ds['instance_ids'], ds['img_paths'], ds['captions'],
                                    test_preprocess)
        ds = val_dataset.gallery
        val_gallery_set = GalleryDataset(ds['instance_ids'],
                                    ds['img_paths'],
                                    test_preprocess)
        relative_train_dataset = SynCPRDataset(data_path=syncpr_data_path, json_path=json_path, split='train', \
                                        mode='relative', preprocess=train_preprocess, setting=setting)
    else:
        ds = val_dataset.query
        val_query_set = QueryDataset(ds['instance_ids'], ds['img_paths'], ds['captions'],
                                    preprocess)
        ds = val_dataset.gallery
        val_gallery_set = GalleryDataset(ds['instance_ids'],
                                    ds['img_paths'],
                                    preprocess)
        relative_train_dataset = SynCPRDataset(data_path=syncpr_data_path, split='train', \
                                        mode='relative', preprocess=preprocess, setting=setting)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs

    # Define the train dataset and the combining function
    if False:
        sampler = MultiCprBatchSampler(
            dataset=relative_train_dataset,
            batch_size=batch_size,
            max_per_cpr=2,
            drop_last=True
        )

        relative_train_loader = DataLoader(
            relative_train_dataset,
            batch_sampler=sampler,
            num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
        )
    else:
        relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                        num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                        drop_last=True, shuffle=True)    

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
                        'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_arithmetic_by_epoch = 0
        best_arithmetic_by_step = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame_by_epoch = pd.DataFrame()
    validation_log_frame_by_step = pd.DataFrame()
    
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions, query_ids) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            blip_model.train()
            
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions, "query_id":query_ids})
                loss = 0.
                for key in loss_dict.keys():
                    if key == 'loss_fda':
                        # FDA loss is always weighted with 1.0 (main loss)
                        loss += loss_dict[key]
                    elif key == 'loss_fd':
                        # Feature Diversity loss weighted by lambda_1
                        loss += kwargs.get('loss_fd', 1.0) * loss_dict[key]
                    elif key == 'loss_mfr':
                        # Masked Feature Reasoning loss weighted by lambda_2
                        loss += kwargs.get('loss_mfr', 0.5) * loss_dict[key]
                    else:
                        # Other losses (backward compatibility)
                        loss += kwargs.get(key, 1.0) * loss_dict[key]
            
            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

            if step % validation_step == 0 and step != 0:
                blip_model.eval()
                # extract target image features
                
                R1, R5, R10, mAP = compute_ticpr_val_metrics(blip_model, val_query_set, val_gallery_set, txt_processors, soft=True)
                # group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
                results_dict = {
                    'recall_at1': float(R1),
                    'recall_at5': float(R5),
                    'recall_at10': float(R10),
                    'mAP': float(mAP),
                }
                print(json.dumps(results_dict, indent=4))

                # Validation CSV logging
                log_dict = {'step': int(step)}
                log_dict.update(results_dict)
                validation_log_frame_by_step = pd.concat([validation_log_frame_by_step, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame_by_step.to_csv(str(training_path / 'validation_metrics_by_step.csv'), index=False)
                if save_training:
                    if save_best and results_dict['recall_at1'] > best_arithmetic_by_step:
                        best_arithmetic_by_step = results_dict['recall_at1']
                        save_model('tuned_recall_at1_step', step, blip_model, training_path, mode='step')

        loss_log_dict = {'epoch': int(epoch)}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            # extract target image features
            
            R1, R5, R10, mAP = compute_ticpr_val_metrics(blip_model, val_query_set, val_gallery_set, txt_processors, soft=True)
            results_dict = {
                'recall_at1': float(R1),
                'recall_at5': float(R5),
                'recall_at10': float(R10),
                'mAP': float(mAP),
            }
            print(json.dumps(results_dict, indent=4))
            # Validation CSV logging
            log_dict = {'epoch': int(epoch)}
            log_dict.update(results_dict)
            validation_log_frame_by_epoch = pd.concat([validation_log_frame_by_epoch, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame_by_epoch.to_csv(str(training_path / 'validation_metrics_by_epoch.csv'), index=False)

            if save_training:
                if save_best and results_dict['recall_at1'] > best_arithmetic_by_epoch:
                    best_arithmetic_by_epoch = results_dict['recall_at1']
                    save_model('tuned_recall_at1_epoch', epoch, blip_model, training_path, mode='epoch')
                if save_last:
                    save_model('tuned_last_recall_at1', epoch, blip_model, training_path, mode='epoch')

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--syncpr-data-path", type=str, default="/mnt/cache/SynCPR_data", help="Path to SynCPR dataset")
    parser.add_argument("--itcpr-root", type=str, default="/mnt/cache/ITCPR", help="Root path for ITCPR dataset")
    parser.add_argument("--json-path", type=str, default="processed_train_new.json")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--exp-name", default="cir", type=str, help="name of the running experiment")
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str, help="[blip2_cir_cat, blip2_cir]")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--resume-path", type=str, default=None, help="resuming path for the current run")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--setting", nargs='+', default=['omini'], type=str, help="CIRHS setting")

    # loss setting (aligned with FAFA paper)
    parser.add_argument("--loss-fda", default=1.0, type=float, help="Fine-grained Dynamic Alignment loss weight")
    parser.add_argument("--loss-fd", default=1.0, type=float, help="Feature Diversity loss weight (lambda_1 in paper)")
    parser.add_argument("--loss-mfr", default=0.5, type=float, help="Masked Feature Reasoning loss weight (lambda_2 in paper)")
    parser.add_argument("--fda-k", default=6, type=int, help="Number of selected fine-grained features in FDA")
    parser.add_argument("--fda-alpha", default=0.5, type=float, help="Soft label strength in FDA")
    parser.add_argument("--fd-margin", default=0.5, type=float, help="Margin parameter m in Feature Diversity loss")
    
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--validation-step", default=1e6, type=int, help="Validation frequency expressed in steps")

    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save the best model during training")
    parser.add_argument("--save-last", dest="save_last", action='store_true',
                        help="Save the last model during training")
    parser.add_argument("--save-memory", dest="save_memory", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() != 'cpr':
        raise ValueError("Dataset should be 'CPR'")
    print(f"save-memory: {args.save_memory}")
    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "exp_name": args.exp_name,
        "blip_model_name": args.blip_model_name,
        "setting": args.setting,
        "backbone": args.backbone,
        "resume_path": args.resume_path,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "validation_step": args.validation_step,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "save_last": args.save_last,
        "syncpr_data_path": args.syncpr_data_path,
        "itcpr_root": args.itcpr_root,
        "loss_fda": args.loss_fda,
        "loss_fd": args.loss_fd,
        "loss_mfr": args.loss_mfr,
        "fda_k": args.fda_k,
        "fda_alpha": args.fda_alpha,
        "fd_margin": args.fd_margin,
        "save_memory": args.save_memory,
        "json_path": args.json_path
    }
    if args.dataset.lower() == 'cpr':
        clip_finetune_cpr(**training_hyper_params)
    else:
        raise ValueError("Only CPR dataset is supported")


