import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_losses(json_paths, dataset_names, output_dir=None):
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.suptitle('Training Curves for Context Encoder Across Datasets')
    
    # Colors for different datasets
    colors = ['b', 'r', 'g']
    
    # Plot generator and discriminator losses for each dataset
    for json_path, dataset_name, color in zip(json_paths, dataset_names, colors):
        with open(json_path, 'r') as f:
            history = json.load(f)
        
        epochs = [entry['epoch'] for entry in history['epoch_losses']]
        generator_losses = [entry['generator_loss'] for entry in history['epoch_losses']]
        discriminator_losses = [entry['discriminator_loss'] for entry in history['epoch_losses']]
        
        # Plot generator losses
        ax1.plot(epochs, generator_losses, f'{color}-', 
                label=f'{dataset_name} Generator', marker='o', markersize=4)
        
        # Plot discriminator losses
        ax2.plot(epochs, discriminator_losses, f'{color}-', 
                label=f'{dataset_name} Discriminator', marker='o', markersize=4)
    
    # Configure subplots
    ax1.set_title('Generator Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('Discriminator Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_losses.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training losses from multiple JSON history files')
    parser.add_argument('--json_paths', type=str, nargs='+',
                      default=['outputs_celeba/training_history.json', 'outputs_cityscapes/training_history.json', 'outputs_places365/training_history.json'],
                      help='Paths to the training history JSON files (space-separated)')
    parser.add_argument('--dataset_names', type=str, nargs='+',
                      default=['CelebA', 'Cityscapes', 'Places365'],
                      help='Names of the datasets (space-separated)')
    parser.add_argument('--output_dir', type=str, default='plots',
                      help='Directory to save the plot')
    args = parser.parse_args()
    
    if len(args.json_paths) != len(args.dataset_names):
        raise ValueError("Number of JSON paths must match number of dataset names")
    
    plot_training_losses(args.json_paths, args.dataset_names, args.output_dir)

if __name__ == '__main__':
    main() 