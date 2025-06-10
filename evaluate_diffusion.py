import os
import torch
import torch.nn as nn
from PIL import Image
from evaluate_models import calculate_psnr
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from diffusers import DiffusionPipeline
from data import InpaintingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_combined_images(path):
    for i in range(5):
        masked_img = Image.open(os.path.join(path, f"masked_{i:04d}.png")).convert('RGB')
        og_img = Image.open(os.path.join(path, f"original_{i:04d}.png")).convert('RGB')
        inpainted_img = Image.open(os.path.join(path, f"inpainted_{i:04d}.png")).convert('RGB')
        inpainted_img = inpainted_img.resize(masked_img.size)
        
        total_width = masked_img.width + og_img.width + inpainted_img.width
        height = masked_img.height

        combined = Image.new("RGB", (total_width, height))
        x_offset = 0
        for img in [og_img, masked_img, inpainted_img]:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width

        # Save
        combined.save(os.path.join('gen_outputs', f"diffusion_cityscape_{i:01d}.png"))


def save_grid():
    im1 = Image.open('gen_outputs/diffusion_cityscape_0.png').convert('RGB')
    dst = Image.new('RGB', (im1.width, int(5 * im1.height)))
    for i in range(5):
        img = Image.open(f'gen_outputs/diffusion_cityscape_{i:01d}.png').convert('RGB')
        dst.paste(img, (0, int(i * im1.height)))
    dst.save('gen_outputs/diffusion_citiscape_grid.png')


def calc_metrics(pipe, device, path, mask_type='square', mask_size=64, num_samples=100):
    dataset = InpaintingDataset(root=path, mask_type=mask_type)
    loader = DataLoader(dataset, batch_size=1, 
                        pin_memory=True, shuffle=False, persistent_workers=True, 
                        num_workers=24)
    
    transform = transforms.ToTensor()
    
    l1_loss_fn = nn.L1Loss()
    l2_loss_fn = nn.MSELoss()
    
    total_l1_loss = 0
    total_l2_loss = 0
    total_psnr = 0

    for i, (masked_image, img, mask) in enumerate(loader):
        img = img.to(device)
        mask = mask.to(device)
        result = pipe(prompt="", image=img, mask_image=mask).images[0]
        result = result.resize((128,128))
        result = transform(result).to(device)

        masked_pred = result * mask
        masked_orig = img * mask
    
        l1_loss = l1_loss_fn(masked_pred, masked_orig)
        l2_loss = l2_loss_fn(masked_pred, masked_orig)
        psnr = calculate_psnr(masked_pred, masked_orig)

        total_l1_loss += l1_loss.item()
        total_l2_loss += l2_loss.item()
        total_psnr += psnr
        
        if i == num_samples - 1:
            break

    num_images = num_samples
    avg_l1_loss = total_l1_loss / num_images
    avg_l2_loss = total_l2_loss / num_images
    avg_psnr = total_psnr / num_images
    
    return {
        'l1_loss': avg_l1_loss,
        'l2_loss': avg_l2_loss,
        'psnr': avg_psnr,
        'num_images': num_images
    }


def all_datasets_metrics(paths, num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to(device)
    mask_type = 'square'
    mask_size = 64
    image_size = 128
    output_file = "diffusion_eval.txt"

    with open(output_file, 'w') as f:
        f.write("Diffusion Model Evaluation Results\n")
        f.write("=======================\n\n")
        f.write(f"Mask Type: {mask_type}\n")
        f.write(f"Mask Size: {mask_size}\n")
        f.write(f"Image Size: {image_size}\n\n")
    
    for key, value in paths.items():
        print(f"Calculating metrics for {key} dataset")
        loss_dict = calc_metrics(pipe, device, value, mask_type=mask_type, mask_size=mask_size, num_samples=num_samples)

        with open(output_file, 'a') as f:
            f.write(f"{key}:\n")
            f.write(f"Mean L1 Loss: {loss_dict['l1_loss']:.4f}\n")
            f.write(f"Mean L2 Loss: {loss_dict['l2_loss']:.4f}\n")
            f.write(f"PSNR: {loss_dict['psnr']:.2f} dB\n")
            f.write(f"Number of Images: {loss_dict['num_images']}\n\n")

    
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    # path = 'cityscapes_inpainted_outputs' #'celeba_inpainted_outputs'
    # print(path)
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # save_combined_images(path)
    # save_grid()
    # print(calc_metrics(path, transform))
    paths = {
        'CelebA-HQ': 'data/celeba_hq_256',
        'Cityscapes': 'data/cityscapes/val/img'
    }
    print(paths)
    print(all_datasets_metrics(paths, num_samples=20))
    
