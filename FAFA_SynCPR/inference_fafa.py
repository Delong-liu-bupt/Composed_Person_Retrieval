#!/usr/bin/env python
"""
FAFA Model Inference Script
Performs inference using a trained FAFA model for Composed Person Retrieval
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from data_utils import targetpad_transform, squarepad_transform, squarepad_transform_test, SynCPRDataset, ITCPRDataset, QueryDataset, GalleryDataset
from lavis.models import registry
from lavis.models import load_model_and_preprocess
from validate_blip import compute_ticpr_val_metrics


def load_model_from_checkpoint(checkpoint_path, model_name='blip2_fafa_cpr', device='cuda'):
    """Load FAFA model from checkpoint file"""
    print(f"Loading model from {checkpoint_path}")

    # Load model and preprocessors
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=model_name,
        model_type='pretrain',
        is_eval=True,
        device=device
    )

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("✓ Checkpoint loaded successfully")
    except RuntimeError as e:
        if "PytorchStreamReader failed" in str(e):
            print(f"\n⚠️ WARNING: Model file appears to be corrupted: {checkpoint_path}")
            print("This can happen if the training was interrupted during saving.")
            print("\nPossible solutions:")
            print("1. Try loading a different checkpoint (e.g., tuned_recall_at1_epoch.pt)")
            print("2. Resume training from an earlier checkpoint")
            print("3. Re-train the model")
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        else:
            raise

    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, txt_processors


def main():
    parser = argparse.ArgumentParser(description='FAFA Model Inference')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--model-name', type=str, default='tuned_recall_at1_step.pt',
                        help='Model checkpoint filename in saved_models directory')
    parser.add_argument('--dataset', type=str, default='itcpr', choices=['itcpr'],
                        help='Dataset to evaluate on (ITCPR)')
    parser.add_argument('--itcpr-root', type=str, default='/mnt/cache/liudelong/data',
                        help='Root path for ITCPR dataset')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Paths
    exp_path = Path(args.exp_dir)
    hyperparams_path = exp_path / 'training_hyperparameters.json'
    model_path = exp_path / 'saved_models' / args.model_name

    # Check paths exist
    if not exp_path.exists():
        raise ValueError(f"Experiment directory not found: {exp_path}")
    if not hyperparams_path.exists():
        raise ValueError(f"Hyperparameters file not found: {hyperparams_path}")
    if not model_path.exists():
        raise ValueError(f"Model checkpoint not found: {model_path}")

    print("=" * 80)
    print(f"FAFA Model Inference")
    print("=" * 80)
    print(f"Experiment: {exp_path.name}")
    print(f"Model: {args.model_name}")

    # Load hyperparameters
    print("\nLoading training hyperparameters...")
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)

    # Extract FDA parameters
    fda_k = hyperparams.get('fda_k', 6)
    fda_alpha = hyperparams.get('fda_alpha', 0.5)
    fd_margin = hyperparams.get('fd_margin', 0.5)
    use_soft = True  # Using soft similarity aggregation

    print(f"\nFDA Parameters:")
    print(f"  - FDA k (top-k features): {fda_k}")
    print(f"  - FDA alpha (soft label strength): {fda_alpha}")
    print(f"  - FD margin: {fd_margin}")

    # Setup device
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    # Load model
    print(f"\nLoading model from {model_path}...")
    model, txt_processors = load_model_from_checkpoint(model_path, device=device)

    # Set FDA parameters on model
    if hasattr(model, 'fda_k'):
        model.fda_k = fda_k
        model.fda_alpha = fda_alpha
        print(f"Set model FDA k={fda_k}, alpha={fda_alpha}")
    if hasattr(model, 'use_soft'):
        model.use_soft = use_soft
        print(f"Set model to use soft similarity aggregation")

    # Setup text processor is already loaded
    print("\nUsing loaded text processor...")

    # Setup transform (use same as training)
    transform = hyperparams.get('transform', 'squarepad')
    if transform == 'squarepad':
        # Use test transform to match training
        preprocess = squarepad_transform_test(224)
        print("Using squarepad_transform_test (matches training)")
    else:
        preprocess = targetpad_transform(1.25, 224)
        print("Using targetpad_transform")

    # Load datasets
    print(f"\nLoading ITCPR validation dataset...")

    # Load ITCPR validation dataset (same as training validation)
    val_dataset = ITCPRDataset(root=args.itcpr_root)
    ds = val_dataset.query
    val_query_set = QueryDataset(
        ds['instance_ids'],
        ds['img_paths'],
        ds['captions'],
        preprocess
    )
    print(f"Query set size: {len(val_query_set)}")

    ds = val_dataset.gallery
    val_gallery_set = GalleryDataset(
        ds['instance_ids'],
        ds['img_paths'],
        preprocess
    )
    print(f"Gallery set size: {len(val_gallery_set)}")

    # Run evaluation
    print("\n" + "=" * 80)
    print("Running inference on validation set...")
    print("=" * 80)

    with torch.no_grad():
        R1, R5, R10, mAP = compute_ticpr_val_metrics(
            model,
            val_query_set,
            val_gallery_set,
            txt_processors,
            soft=use_soft
        )

    # Print results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    print(f"Dataset: ITCPR (In-the-wild Test for CPR)")
    print(f"Model: {args.model_name}")
    print(f"FDA Parameters: k={fda_k}, alpha={fda_alpha}")
    print("-" * 40)
    print(f"Recall@1:  {R1:.4f}")
    print(f"Recall@5:  {R5:.4f}")
    print(f"Recall@10: {R10:.4f}")
    print(f"mAP:       {mAP:.4f}")
    print("=" * 80)

    # Save results
    results_path = exp_path / f'inference_results_{args.model_name}.json'
    results = {
        'model': args.model_name,
        'dataset': args.dataset,
        'fda_k': fda_k,
        'fda_alpha': fda_alpha,
        'fd_margin': fd_margin,
        'recall_at_1': float(R1),
        'recall_at_5': float(R5),
        'recall_at_10': float(R10),
        'mAP': float(mAP),
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()