import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import InpaintingDataset
from model import ContextEncoder, Discriminator

def train(args):
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = InpaintingDataset(root=args.data_root, dataset=args.dataset,
                                image_size=args.image_size, mask_size=args.mask_size, 
                                mask_type=args.mask_type)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    G = ContextEncoder().to(device)
    D = Discriminator().to(device)

    optim_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        epoch_loss_G, epoch_loss_D = 0.0, 0.0
        for masked, img, mask in tqdm(loader, desc=f"Epoch {epoch}"):
            masked, img, mask = masked.to(device), img.to(device), mask.to(device)
            optim_D.zero_grad()
            with torch.no_grad():
                pred = G(masked)
                comp = img * (1 - mask) + pred * mask
            real_logits = D(img)
            fake_logits = D(comp)
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            loss_D = bce(real_logits, real_labels) + bce(fake_logits, fake_labels)
            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            pred = G(masked)
            comp = img * (1 - mask) + pred * mask
            logits = D(comp)
            adv_labels = torch.ones_like(logits)
            loss_adv = bce(logits, adv_labels)
            loss_rec = mse(pred * mask, img * mask)
            loss_G = args.lambda_rec * loss_rec + args.lambda_adv * loss_adv
            loss_G.backward()
            optim_G.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

        avg_G = epoch_loss_G / len(loader)
        avg_D = epoch_loss_D / len(loader)
        print(f"Epoch {epoch} | Generator Loss: {avg_G:.4f} | Discriminator Loss: {avg_D:.4f}")

        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(args.out_dir, f"G_epoch{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, f"D_epoch{epoch}.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Context Encoder Inpainting Training')
    parser.add_argument('--data_root', type=str, default='/home/ajj/Dev/context_encoder/data/celebahq-256/celeba_hq_256/', help='path to dataset root')
    parser.add_argument('--dataset', type=str, default='celeba', choices=['celeba'], help='dataset to use')
    parser.add_argument('--image_size', type=int, default=128, help='size of input images')
    parser.add_argument('--mask_size', type=int, default=64, help='size of square mask')
    parser.add_argument('--mask_type', type=str, default='mixed', 
                       choices=['square', 'circle', 'triangle', 'ellipse', 'irregular', 'random_patches', 'mixed'],
                       help='type of mask to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lambda_rec', type=float, default=0.999, help='weight for reconstruction loss')
    parser.add_argument('--lambda_adv', type=float, default=0.001, help='weight for adversarial loss')
    parser.add_argument('--gpu', action='store_true', help='use GPU if available')
    parser.add_argument('--out_dir', type=str, default='outputs', help='output directory for checkpoints')
    args = parser.parse_args()
    train(args)
