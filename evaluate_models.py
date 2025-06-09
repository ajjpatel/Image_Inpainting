import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import random
from model import ContextEncoder
from data import InpaintingDataset

def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def get_all_image_files(root_dir):
    """Get all image files recursively from a directory and its subdirectories."""
    image_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def evaluate_model(model, val_dir, device, transform, mask_type='square', mask_size=64, image_size=128, num_samples=1000):
    """Evaluate model on validation images with masking."""
    model.eval()
    l1_loss_fn = nn.L1Loss()
    l2_loss_fn = nn.MSELoss()
    
    total_l1_loss = 0
    total_l2_loss = 0
    total_psnr = 0
    
    # Create dataset with masking
    dataset = InpaintingDataset(
        root=val_dir,
        dataset='celeba',  # This will be ignored for validation
        image_size=image_size,
        mask_size=mask_size,
        mask_type=mask_type
    )
    
    # Get all image files recursively from validation directory
    image_files = get_all_image_files(val_dir)
    
    if not image_files:
        print(f"Warning: No images found in {val_dir}")
        return {
            'l1_loss': float('inf'),
            'l2_loss': float('inf'),
            'psnr': 0.0,
            'num_images': 0
        }
    
    # Randomly sample num_samples images
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    print(f"Evaluating on {len(image_files)} images...")
    
    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Evaluating images"):
            # Load and preprocess image
            img = Image.open(img_file).convert('RGB')
            img = transform(img)
            
            # Get masked image and mask
            masked_img, orig_img, mask = dataset[random.randint(0, len(dataset)-1)]
            masked_img = masked_img.unsqueeze(0).to(device)
            orig_img = orig_img.to(device)
            mask = mask.to(device)
            
            # Get model prediction
            pred = model(masked_img)
            
            # Calculate metrics only on masked regions
            masked_pred = pred.squeeze(0) * mask
            masked_orig = orig_img * mask
            
            l1_loss = l1_loss_fn(masked_pred, masked_orig)
            l2_loss = l2_loss_fn(masked_pred, masked_orig)
            psnr = calculate_psnr(masked_pred, masked_orig)
            
            total_l1_loss += l1_loss.item()
            total_l2_loss += l2_loss.item()
            total_psnr += psnr
    
    # Calculate averages
    num_images = len(image_files)
    avg_l1_loss = total_l1_loss / num_images
    avg_l2_loss = total_l2_loss / num_images
    avg_psnr = total_psnr / num_images
    
    return {
        'l1_loss': avg_l1_loss,
        'l2_loss': avg_l2_loss,
        'psnr': avg_psnr,
        'num_images': num_images
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on validation datasets')
    parser.add_argument('--model_paths', type=str, nargs='+',
                      default=['outputs_celeba/G_best.pth', 
                              'outputs_cityscapes/G_best.pth', 
                              'outputs_places365/G_best.pth'],
                      help='Paths to the trained model checkpoints')
    parser.add_argument('--val_dirs', type=str, nargs='+',
                      default=['/home/ajj/Dev/context_encoder/data/celebahq-256/celeba_hq_256/0',
                              '/home/ajj/Dev/context_encoder/data/cityscapes/val/img',
                              '/home/ajj/Dev/context_encoder/data/places365/val/normal'],
                      help='Paths to validation image directories')
    parser.add_argument('--dataset_names', type=str, nargs='+',
                      default=['CelebA', 'Cityscapes', 'Places365'],
                      help='Names of the datasets')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt',
                      help='File to save evaluation results')
    parser.add_argument('--gpu', type=int, default=None,
                      help='GPU ID to use. If not specified, CPU will be used.')
    parser.add_argument('--mask_type', type=str, default='square',
                      choices=['square', 'circle', 'triangle', 'ellipse', 'irregular', 'random_patches'],
                      help='Type of mask to use for evaluation')
    parser.add_argument('--mask_size', type=int, default=64,
                      help='Size of the mask')
    parser.add_argument('--image_size', type=int, default=128,
                      help='Size of the input images')
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load models and evaluate
    results = {}
    for model_path, val_dir, dataset_name in zip(args.model_paths, args.val_dirs, args.dataset_names):
        print(f"\nEvaluating {dataset_name}...")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping {dataset_name}.")
            continue
            
        # Check if validation directory exists
        if not os.path.exists(val_dir):
            print(f"Warning: Validation directory {val_dir} not found. Skipping {dataset_name}.")
            continue
        
        # Create model and load state dict
        model = ContextEncoder(input_channels=3, feature_dim=512)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        if args.gpu is not None and torch.cuda.is_available():
            model = model.to(device)
        
        # Evaluate
        metrics = evaluate_model(model, val_dir, device, transform, 
                               mask_type=args.mask_type,
                               mask_size=args.mask_size,
                               image_size=args.image_size)
        results[dataset_name] = metrics
        
        if metrics['num_images'] > 0:
            print(f"{dataset_name} Results:")
            print(f"Mean L1 Loss: {metrics['l1_loss']:.4f}")
            print(f"Mean L2 Loss: {metrics['l2_loss']:.4f}")
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            print(f"Number of Images: {metrics['num_images']}")
        else:
            print(f"Skipping {dataset_name} due to no valid images found.")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=======================\n\n")
        f.write(f"Mask Type: {args.mask_type}\n")
        f.write(f"Mask Size: {args.mask_size}\n")
        f.write(f"Image Size: {args.image_size}\n\n")
        for dataset_name, metrics in results.items():
            f.write(f"{dataset_name}:\n")
            if metrics['num_images'] > 0:
                f.write(f"Mean L1 Loss: {metrics['l1_loss']:.4f}\n")
                f.write(f"Mean L2 Loss: {metrics['l2_loss']:.4f}\n")
                f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
                f.write(f"Number of Images: {metrics['num_images']}\n\n")
            else:
                f.write("No valid images found for evaluation\n\n")
    
    print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    main() 