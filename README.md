# Image Inpainting using Context Encoders and Diffusion Model

Contributors: Mansi Nanavati, Ajjkumar Patel

This projects tackles the task of Image Inpainting with two different methods: Context Encoders and Stable Diffusion. The pipelines are tested on three datasets: CelebA-HQ, Cityscapes and Places365.

The report and presentation slides are saved as `ECE_285_Image_Inpainting_Project.pdf` and `Image inpainting slides.pdf` respectively. Link to the slides: https://docs.google.com/presentation/d/1pqPN4j-e1wX6Z7c7IR5j7GyM2fOrUS3oVqVOkFf41UE/edit?usp=sharing

## Context Encoder
To train the context encoder, use the following command 
``` 
python train.py --data_root <path_to_dataset> --dataset <dataset_name> --epochs 20 --mask_type mixed --gpu
```
The final generator model will be saved as `outputs/G_best.pth`
For inference, run the following command:
```
python generate_inpainted_samples.py --comparison_grid --mask_type "all" --mask_size 96 --data_root "data/cityscapes/val/img" --gpu
```
The comparison grid is saved as `combined_comparison_grid.png`. To calculate the metrics, run `evaluate_models.py`
## Stable Diffusion
To run the pipeline, head to the `diffusion_inpainting.ipynb` and run all the cells. The inpainted images are saved in `{dataset}_inpainted_outputs/`. The comparison grid will be saved in `gen_outputs/diffusion_{dataset}_grid.png`.
To calculate metrics, run `evaluate_diffusion.py`.
