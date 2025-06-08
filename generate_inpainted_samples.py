import os
import argparse
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
from data import InpaintingDataset
from model import ContextEncoder


def denorm(img_tensor):
    return img_tensor * 0.5 + 0.5

def save_triplet(orig, masked, repainted, out_path, mask_type=None):
    orig_img = denorm(orig).cpu().numpy().transpose(1,2,0)
    masked_img = denorm(masked).cpu().numpy().transpose(1,2,0)
    repainted_img = denorm(repainted).cpu().numpy().transpose(1,2,0)
    triplet = np.concatenate([orig_img, masked_img, repainted_img], axis=1)
    triplet = (triplet * 255).astype(np.uint8)
    
    triplet_pil = Image.fromarray(triplet)
    triplet_pil.save(out_path)

def save_comparison_grid(samples_by_type, output_path):
    if not samples_by_type:
        return
    
    from PIL import Image
    
    sample_height = list(samples_by_type.values())[0].shape[0]
    sample_width = list(samples_by_type.values())[0].shape[1]
    
    mask_types = list(samples_by_type.keys())
    grid_height = len(mask_types) * sample_height
    grid_width = sample_width
    
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, (_, triplet_array) in enumerate(samples_by_type.items()):
        triplet_pil = Image.fromarray(triplet_array)
        
        y_offset = i * sample_height
        grid_img.paste(triplet_pil, (0, y_offset))
    
    grid_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate inpainted samples using ContextEncoder with various mask types")
    parser.add_argument('--data_root', type=str, default="data/celebahq-256/celeba_hq_256/", help='Path to dataset root')
    parser.add_argument('--model_path', type=str, default="outputs/G_epoch1.pth", help='Path to trained ContextEncoder .pth file')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of generations to dump')
    parser.add_argument('--image_size', type=int, default=128, help='Image size (should match training)')
    parser.add_argument('--mask_size', type=int, default=64, help='Mask size (should match training)')
    parser.add_argument('--mask_type', type=str, default='mixed', 
                       choices=['square', 'circle', 'triangle', 'ellipse', 'irregular', 'random_patches', 'mixed', 'all'],
                       help='Type of mask to use for generation. Use "all" to generate samples with all mask types')
    parser.add_argument('--output_dir', type=str, default='gen_outputs', help='Directory to save output images')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--comparison_grid', action='store_true', help='Generate comparison grid showing all mask types for same images')
    parser.add_argument('--save_individual', action='store_true', default=True, help='Save individual triplet images')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    model = ContextEncoder().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    if args.mask_type == 'all':
        mask_types = ['square', 'circle', 'triangle', 'ellipse', 'irregular', 'random_patches']
    else:
        mask_types = [args.mask_type]

    print(f"Generating samples with mask types: {mask_types}")

    if args.comparison_grid and len(mask_types) > 1:
        print("Generating comparison grids...")
        
        for sample_idx in tqdm(range(args.num_samples), desc="Generating comparison grids"):
            np.random.seed(sample_idx + 42)
            
            samples_by_type = {}
            base_img_idx = np.random.randint(0, 1000)
            
            for mask_type in mask_types:    
                dataset = InpaintingDataset(
                    root=args.data_root, 
                    dataset='celeba', 
                    image_size=args.image_size, 
                    mask_size=args.mask_size,
                    mask_type=mask_type
                )
                
                masked_img, orig_img, mask = dataset[base_img_idx % len(dataset)]
                masked_img = masked_img.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(masked_img)
                
                mask = mask.to(device)
                repainted = orig_img.to(device) * (1 - mask) + pred.squeeze(0) * mask
                
                orig_img_np = denorm(orig_img).cpu().numpy().transpose(1,2,0)
                masked_img_np = denorm(masked_img.squeeze(0)).cpu().numpy().transpose(1,2,0)
                repainted_img_np = denorm(repainted).cpu().numpy().transpose(1,2,0)
                triplet = np.concatenate([orig_img_np, masked_img_np, repainted_img_np], axis=1)
                triplet = (triplet * 255).astype(np.uint8)
                
                samples_by_type[mask_type] = triplet
            
            grid_path = os.path.join(args.output_dir, f'comparison_grid_{sample_idx+1:03d}.png')
            save_comparison_grid(samples_by_type, grid_path)
    
    else:
        print("Generating individual samples...")
        
        for mask_type in mask_types:
            print(f"Generating samples with {mask_type} masks...")
            
            dataset = InpaintingDataset(
                root=args.data_root, 
                dataset='celeba', 
                image_size=args.image_size, 
                mask_size=args.mask_size,
                mask_type=mask_type
            )
            
            mask_samples = args.num_samples if len(mask_types) == 1 else max(1, args.num_samples // len(mask_types))
            
            for i in tqdm(range(mask_samples), desc=f"Generating {mask_type} samples"):
                idx = np.random.randint(0, len(dataset))
                masked_img, orig_img, mask = dataset[idx]
                masked_img = masked_img.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(masked_img)

                mask = mask.to(device)
                repainted = orig_img.to(device) * (1 - mask) + pred.squeeze(0) * mask
                
                if args.save_individual:
                    if len(mask_types) == 1:
                        out_path = os.path.join(args.output_dir, f'sample_{i+1:03d}.png')
                    else:
                        out_path = os.path.join(args.output_dir, f'sample_{mask_type}_{i+1:03d}.png')
                    
                    save_triplet(orig_img, masked_img.squeeze(0), repainted, out_path, mask_type)

    print(f"\nGeneration complete!")
    print(f"Output directory: {args.output_dir}")
    
    if args.comparison_grid and len(mask_types) > 1:
        print(f"Generated {args.num_samples} comparison grids showing all mask types")
    else:
        total_samples = args.num_samples if len(mask_types) == 1 else args.num_samples * len(mask_types)
        print(f"Generated {total_samples} individual samples with mask types: {mask_types}")
    
    print(f"Mask types used: {', '.join(mask_types)}")
    print(f"Model: {args.model_path}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Mask size: {args.mask_size}")

if __name__ == '__main__':
    main() 