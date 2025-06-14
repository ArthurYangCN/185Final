import torch
import os

def get_model_info():
    """Extract model information from checkpoints"""
    
    print("=== Model Architecture Information ===")
    
    # DC-GAN info
    dcgan_files = [f for f in os.listdir('checkpoints/') if f.startswith('dcgan_generator')]
    if dcgan_files:
        dcgan_path = f"checkpoints/{dcgan_files[0]}"
        checkpoint = torch.load(dcgan_path, map_location='cpu')
        
        # Extract ngf from first layer
        ngf = checkpoint['main.0.weight'].shape[1] // 8
        nz = checkpoint['main.0.weight'].shape[0]  # latent vector size
        
        print(f"DC-GAN Parameters:")
        print(f"  - Latent vector size (nz): {nz}")
        print(f"  - Generator features (ngf): {ngf}")
        print(f"  - Number of layers: 4 (3 ConvTranspose + 1 output)")
        
        # Count total parameters
        total_params = sum(p.numel() for p in checkpoint.values())
        print(f"  - Total parameters: {total_params:,}")
    
    # VAE info
    vae_files = [f for f in os.listdir('checkpoints/') if f.startswith('vae_epoch')]
    if vae_files:
        vae_path = f"checkpoints/{vae_files[0]}"
        checkpoint = torch.load(vae_path, map_location='cpu')
        
        # Extract latent_dim
        latent_dim = checkpoint['fc_mu.weight'].shape[0]
        encoder_features = checkpoint['encoder.4.weight'].shape[0]  # Last conv layer
        
        print(f"\nVAE Parameters:")
        print(f"  - Latent dimension: {latent_dim}")
        print(f"  - Encoder features: {encoder_features}")
        print(f"  - Architecture: 3 Conv layers + 3 ConvTranspose layers")
        
        # Count total parameters
        total_params = sum(p.numel() for p in checkpoint.values())
        print(f"  - Total parameters: {total_params:,}")
    
    print(f"\n=== Training Information ===")
    print("Based on your training scripts:")
    print("DC-GAN:")
    print("  - Batch size: 256")
    print("  - Learning rate: 0.0002") 
    print("  - Optimizer: Adam (β1=0.5)")
    print("  - Training epochs: 10")
    print("  - Dataset: CIFAR-10 subset (10,000 samples)")
    
    print("VAE:")
    print("  - Batch size: 256")
    print("  - Learning rate: 0.001")
    print("  - Optimizer: Adam")
    print("  - Training epochs: 10") 
    print("  - Beta parameter: 1.0")
    print("  - Dataset: CIFAR-10 subset (10,000 samples)")

if __name__ == "__main__":
    get_model_info()