import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

#
#asked Claude.ai how to create 3*3 config
#
def create_best_comparison_figure():
    """Create the most impactful single figure for your report"""
    
    # Create a 3x3 grid: Real vs DC-GAN vs VAE
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle('CIFAR-10 Generation Comparison: Real vs DC-GAN vs VAE', fontsize=16, y=0.98)
    
    # Row labels
    row_labels = ['Real CIFAR-10', 'DC-GAN Generated', 'VAE Generated']
    
    # Load real CIFAR-10 samples
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        real_batch, _ = next(iter(dataloader))
        
        # Save a grid of real samples
        vutils.save_image(real_batch[:64], 'results/temp_real_grid.png', nrow=8, normalize=True)
    except:
        print("Could not load real CIFAR-10 samples")
    
    # Define the image paths
    image_paths = [
        'results/temp_real_grid.png',  # or any real sample grid you have
        'results/dcgan_evaluation_samples.png',  # Your DC-GAN results
        'results/vae_evaluation_samples.png'  # Your VAE results
    ]
    
    # Alternative paths if evaluation samples don't exist
    alt_paths = [
        'results/real_samples.png',
        'results/dcgan/fake_samples_epoch_9.png',
        'results/vae/generated_epoch_9.png'
    ]
    
    # Plot each row
    for row in range(3):
        # Try primary path first, then alternative
        img_path = image_paths[row]
        if not os.path.exists(img_path) and row < len(alt_paths):
            img_path = alt_paths[row]
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Show full grid in first column
            axes[row, 0].imshow(img)
            axes[row, 0].set_ylabel(row_labels[row], fontsize=12, fontweight='bold')
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            
            # Show zoomed samples in columns 2 and 3
            # Extract individual 32x32 samples from the grid
            if img.shape[0] >= 256:  # If it's a full grid
                # Show top-left sample
                sample1 = img[2:34, 2:34]  # Approximate position
                axes[row, 1].imshow(sample1)
                axes[row, 1].set_xticks([])
                axes[row, 1].set_yticks([])
                
                # Show another sample
                sample2 = img[2:34, 36:68]  # Approximate position
                axes[row, 2].imshow(sample2)
                axes[row, 2].set_xticks([])
                axes[row, 2].set_yticks([])
        else:
            # If image doesn't exist, show text
            for col in range(3):
                axes[row, col].text(0.5, 0.5, f'{row_labels[row]}\n(Run evaluation first)', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
    
    # Column titles
    axes[0, 0].set_title('8×8 Grid View', fontsize=11)
    axes[0, 1].set_title('Sample 1 (Zoomed)', fontsize=11)
    axes[0, 2].set_title('Sample 2 (Zoomed)', fontsize=11)
    
    plt.tight_layout()
    
    # Add caption-style text
    
    plt.savefig('results/figure_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    print("Created figure_comparison.png")
    
    # Also create a simpler version if the above doesn't work well
    create_simple_side_by_side()

def create_simple_side_by_side():
    """Create a simpler side-by-side comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    titles = ['Real CIFAR-10', 'DC-GAN Generated', 'VAE Generated']
    paths = [
        'results/dcgan/fake_samples_epoch_0.png',  # You can change to real samples
        'results/dcgan/fake_samples_epoch_9.png',
        'results/vae/generated_epoch_9.png'
    ]
    
    for i, (title, path) in enumerate(zip(titles, paths)):
        if os.path.exists(path):
            img = mpimg.imread(path)
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'{title}\n(Image not found)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.suptitle('Image Generation Quality Comparison on CIFAR-10', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/figure_simple_comparison.png', dpi=300, bbox_inches='tight')
    print("Created figure_simple_comparison.png")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    create_best_comparison_figure()