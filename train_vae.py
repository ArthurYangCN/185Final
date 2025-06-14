import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import ssl
from models.vae import VAE, vae_loss

# Fixed
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    batch_size = 256
    latent_dim = 64
    num_epochs = 10
    lr = 1e-3
    beta = 1.0# beta-VAE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/vae', exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    subset_size = 10000
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"{len(subset)} samples with batch size {batch_size}")
    print(f"Total batches per epoch: {len(dataloader)}")

    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    recon_losses = []
    kld_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        recon_loss_epoch = 0
        kld_loss_epoch = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            train_loss += loss.item()
            recon_loss_epoch += recon_loss.item()
            kld_loss_epoch += kld_loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(subset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {loss.item() / len(data):.6f} '
                      f'Recon: {recon_loss.item() / len(data):.6f} '
                      f'KLD: {kld_loss.item() / len(data):.6f}')
        
        avg_loss = train_loss / len(subset)
        avg_recon = recon_loss_epoch / len(subset)
        avg_kld = kld_loss_epoch / len(subset)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kld_losses.append(avg_kld)
        
        print(f'Epoch: {epoch} Avg loss: {avg_loss:.4f} '
              f'Recon: {avg_recon:.4f} KLD: {avg_kld:.4f}')
        model.eval()
        with torch.no_grad():
            data_sample = next(iter(dataloader))[0][:64].to(device)
            recon, _, _ = model(data_sample)
            comparison = torch.cat([data_sample[:32], recon[:32]])
            vutils.save_image(comparison.cpu(), 
                            f'results/vae/reconstruction_epoch_{epoch}.png', 
                            nrow=8, normalize=True)
            
            sample = torch.randn(64, latent_dim).to(device)
            generated = model.decode(sample)
            vutils.save_image(generated.cpu(), 
                            f'results/vae/generated_epoch_{epoch}.png', 
                            nrow=8, normalize=True)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/vae_epoch_{epoch}.pth')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 3, 3)
    plt.plot(kld_losses)
    plt.title('KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('results/vae/training_losses.png')
    plt.show()

if __name__ == '__main__':
    main()