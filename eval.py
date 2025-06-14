import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from models.dcgan import Generator
from models.vae import VAE
from PIL import Image

## Asked claude.ai how to write some of the metrics evaluation and fix eval bugs.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_real_cifar_samples(num_samples=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    real_samples = []
    for i, (data, _) in enumerate(dataloader):
        real_samples.append(data)
        if len(real_samples) * 64 >= num_samples:
            break
    
    return torch.cat(real_samples, dim=0)[:num_samples]

def calculate_pixel_diversity(images):
    images_np = images.numpy()
    pixel_std = np.std(images_np, axis=0)
    mean_diversity = np.mean(pixel_std)
    
    return mean_diversity

def calculate_reconstruction_error(model, test_images):
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(test_images.to(device))
        mse = torch.nn.functional.mse_loss(recon, test_images.to(device))
        return mse.item()

def evaluate_models():
    try:
        real_samples = load_real_cifar_samples(500)
        print(f"Loaded {len(real_samples)} real CIFAR-10 samples for comparison")
    except:
        real_samples = None
    
    results = {}
    
    # eval DC-GAN
    dcgan_checkpoint = None
    for f in os.listdir('checkpoints/'):
        if f.startswith('dcgan_generator'):
            dcgan_checkpoint = f"checkpoints/{f}"
            break
    
    if dcgan_checkpoint:
        print(f"Loading: {dcgan_checkpoint}")
        
        checkpoint = torch.load(dcgan_checkpoint, map_location='cpu')
        ngf = checkpoint['main.0.weight'].shape[1] // 8
        print(f"Detected ngf: {ngf}")
        netG = Generator(nz=100, ngf=ngf, nc=3).to(device)
        netG.load_state_dict(torch.load(dcgan_checkpoint, map_location=device))
        netG.eval()
        
        # samples
        with torch.no_grad():
            noise = torch.randn(500, 100, 1, 1, device=device)
            dcgan_samples = netG(noise)
            dcgan_samples = (dcgan_samples + 1) / 2 
        
        # Calculate metrics
        dcgan_diversity = calculate_pixel_diversity(dcgan_samples.cpu())
        
        results['dcgan'] = {
            'diversity': dcgan_diversity,
            'num_samples': len(dcgan_samples)
        }
        
        print(f"Generated {len(dcgan_samples)} samples")
        print(f"Pixel diversity: {dcgan_diversity:.6f}")
        
        vutils.save_image(dcgan_samples[:64], 'results/dcgan_evaluation_samples.png', 
                         nrow=8, normalize=True)
    
    # Eval VAE
    vae_checkpoint = None
    for f in os.listdir('checkpoints/'):
        if f.startswith('vae_epoch'):
            vae_checkpoint = f"checkpoints/{f}"
            break
    
    if vae_checkpoint:
        print(f"Loading: {vae_checkpoint}")
        
        checkpoint = torch.load(vae_checkpoint, map_location='cpu')
        latent_dim = checkpoint['fc_mu.weight'].shape[0]
        print(f"Detected latent_dim: {latent_dim}")
        
        model = VAE(latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(vae_checkpoint, map_location=device))
        model.eval()
        
        with torch.no_grad():
            z = torch.randn(500, latent_dim, device=device)
            vae_samples = model.decode(z)
        vae_diversity = calculate_pixel_diversity(vae_samples.cpu())
        
        # Calculreconstruction error
        if real_samples is not None:
            recon_error = calculate_reconstruction_error(model, real_samples[:100])
            results['vae'] = {
                'diversity': vae_diversity,
                'reconstruction_error': recon_error,
                'num_samples': len(vae_samples)
            }
            print(f"Reconstruction error: {recon_error:.6f}")
        else:
            results['vae'] = {
                'diversity': vae_diversity,
                'num_samples': len(vae_samples)
            }
        
        print(f"Generated {len(vae_samples)} samples")
        print(f"Pixel diversity: {vae_diversity:.6f}")
        
        vutils.save_image(vae_samples[:64], 'results/vae_evaluation_samples.png', 
                         nrow=8, normalize=True)
    
    # Compare results
    print(f"\nSummary")
    if 'dcgan' in results and 'vae' in results:
        print(f"DC-GAN pixel diversity: {results['dcgan']['diversity']:.6f}")
        print(f"VAE pixel diversity: {results['vae']['diversity']:.6f}")
        
        if results['dcgan']['diversity'] > results['vae']['diversity']:
            print("DC-GAN shows higher pixel diversity")
        else:
            print("VAE shows higher pixel diversity")
        
        if 'reconstruction_error' in results['vae']:
            print(f"VAE reconstruction error: {results['vae']['reconstruction_error']:.6f}")
    
    print(f"\nResults")
    dcgan_files = len([f for f in os.listdir('results/dcgan/') if f.endswith('.png')])
    vae_files = len([f for f in os.listdir('results/vae/') if f.endswith('.png')])

    return results

if __name__ == "__main__":
    results = evaluate_models()