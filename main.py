import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms.functional as F

import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

from nets.discriminator import Discriminator
from nets.generator import Generator
from datasets.custom_test import CustomDatasetTest

lr = 0.0002
num_epochs = 5
batch_size = 4
beta1 = 0.5
num_workers = 3
ngpu = 4
patch = 64 # patch size
datapath = "./linked_real_v9"
trainlist = "./filenames/custom_test_real.txt"

simpath = "./linked_sim_v9"
simlist = "./filenames/custom_test_sim.txt"

dataset = CustomDatasetTest(datapath, trainlist)
simset = CustomDatasetTest(simpath, simlist)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
simloader = torch.utils.data.DataLoader(simset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator().to(device)
netG.apply(weights_init)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netD = Discriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

criterion = nn.BCELoss()

# fixed_noise = torch.randn(128, 100, 13, 27, device=device)
fixed_noise = torch.randn(128, 100, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    start = time.time()
    for data, simdata in zip(enumerate(dataloader), enumerate(simloader)):
        i, data = data
        j, simdata = simdata

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()

        real_cpu = data.to(device)
        simdata = simdata.to(device)

        b_size,channels,h,w = real_cpu.shape

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D

        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_real = F.crop(real_cpu, top, left, patch, patch)
        output = torch.mean(netD(cropped_real),dim=(2,3)).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # noise = torch.randn(b_size, 100, 13, 27, device=device)
        # noise = torch.randn(b_size, 100, 1, 1, device=device) # the size of the image (3,256,480)

        # Generate fake image batch with G
        fake = netG(simdata)
        label.fill_(fake_label)

        # Classify all fake batch with D
        cropped_fake = F.crop(fake.detach(), top, left, patch, patch)
        output = torch.mean(netD(cropped_fake),dim=(2,3)).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = torch.mean(netD(fake),dim=(2,3)).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print(time.time()-start)
            start = time.time()

        if (iters%200 ==0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(simdata).detach().cpu().numpy()
            # img = vutils.make_grid(fake, padding=2, normalize=True)

            img = fake[0][0]*255
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('fake'+str(epoch)+'_'+str(i)+'.png')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('lossgraph.png')
