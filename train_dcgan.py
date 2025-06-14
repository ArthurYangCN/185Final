import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
import ssl
from models.dcgan import Generator, Discriminator, weights_init

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    batch_size = 256
    nz = 100
    ngf = 32
    ndf = 32
    num_epochs = 10
    lr = 0.0002
    beta1 = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/dcgan', exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    subset_size = 10000  # only 10k not 50k
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Using {len(subset)} samples with batch size {batch_size}")
    print(f"Total batches per epoch: {len(dataloader)}")
    netG = Generator(nz, ngf, 3).to(device)
    netD = Discriminator(3, ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    G_losses = []
    D_losses = []

    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):
            netD.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            # Train fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            #Generator
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if i % 10 == 0: 
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
        
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            vutils.save_image(fake, f'results/dcgan/fake_samples_epoch_{epoch}.png', 
                            normalize=True, nrow=8)
        
        if epoch % 5 == 0:
            torch.save(netG.state_dict(), f'checkpoints/dcgan_generator_epoch_{epoch}.pth')
            torch.save(netD.state_dict(), f'checkpoints/dcgan_discriminator_epoch_{epoch}.pth')

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('results/dcgan/training_losses.png')
    plt.show()


    main()