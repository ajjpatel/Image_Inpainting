import torch

def make_mask(image_size, mask_size):
    top = torch.randint(0, image_size - mask_size + 1, (1,)).item()
    left = torch.randint(0, image_size - mask_size + 1, (1,)).item()
    mask = torch.zeros((1, image_size, image_size))
    mask[:, top:top+mask_size, left:left+mask_size] = 1
    return mask
