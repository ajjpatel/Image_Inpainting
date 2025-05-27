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

def save_triplet(orig, masked, repainted, out_path):
    orig_img = denorm(orig).cpu().numpy().transpose(1,2,0)
    masked_img = denorm(masked).cpu().numpy().transpose(1,2,0)
    repainted_img = denorm(repainted).cpu().numpy().transpose(1,2,0)
    triplet = np.concatenate([orig_img, masked_img, repainted_img], axis=1)
    triplet = (triplet * 255).astype(np.uint8)
    Image.fromarray(triplet).save(out_path)

def main():
    parser = argparse.ArgumentParser(description="Generate inpainted samples using ContextEncoder")
    parser.add_argument('--data_root', type=str, default="data/celebahq-256/celeba_hq_256/", help='Path to dataset root')
    parser.add_argument('--model_path', type=str, default="outputs/G_epoch1.pth" ,help='Path to trained ContextEncoder .pth file')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of generations to dump')
    parser.add_argument('--image_size', type=int, default=128, help='Image size (should match training)')
    parser.add_argument('--mask_size', type=int, default=64, help='Mask size (should match training)')
    parser.add_argument('--output_dir', type=str, default='gen_outputs', help='Directory to save output images')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    model = ContextEncoder().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = InpaintingDataset(root=args.data_root, dataset='celeba', image_size=args.image_size, mask_size=args.mask_size)

    for i in tqdm(range(args.num_samples)):
        idx = np.random.randint(0, len(dataset))
        masked_img, orig_img, mask = dataset[idx]
        masked_img = masked_img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(masked_img)

        mask = mask.to(device)
        repainted = orig_img.to(device) * (1 - mask) + pred.squeeze(0) * mask
        save_triplet(orig_img, masked_img.squeeze(0), repainted, os.path.join(args.output_dir, f'sample_{i+1:03d}.png'))

    print(f"Done! Saved {args.num_samples} samples to {args.output_dir}")

if __name__ == '__main__':
    main() 