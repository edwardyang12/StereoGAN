import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms.functional as F

import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

from nets.discriminator import OrigDiscriminator, MiniDiscriminator
from nets.generator import ResGenerator as Generator
from utils import weights_init
from datasets.custom_test import CustomDatasetTest
from datasets.custom_dataset import CustomDataset

lr = 0.0002
num_epochs = 15
batch_size = 7
beta1 = 0.5
num_workers = 0
ngpu = 1
patch = 128 # patch size

datapath = "./linked_real_v9"
trainlist = "./filenames/custom_test_real.txt"

simpath = "./linked_sim_v9"
simlist = "./filenames/custom_test_sim.txt"

dataset = CustomDataset(datapath, trainlist)
simset = CustomDataset(simpath, simlist)
simtest = CustomDatasetTest(simpath, simlist)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)
simloader = torch.utils.data.DataLoader(simset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)
simtest = torch.utils.data.DataLoader(simtest, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

netG = Generator().to(device)
# netG.apply(weights_init)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

origD = OrigDiscriminator().to(device)
miniD = MiniDiscriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    origD = nn.DataParallel(origD, list(range(ngpu)))
    miniD = nn.DataParallel(miniD, list(range(ngpu)))
origD.apply(weights_init)
miniD.apply(weights_init)

criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerDorig = optim.Adam(origD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDmini = optim.Adam(miniD.parameters(), lr=lr, betas=(beta1, 0.999))
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
        data, path = data
        j, simdata = simdata
        simdata, simpath = simdata

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        origD.zero_grad()
        miniD.zero_grad()

        real_cpu = data.to(device)
        simdata = simdata.to(device)

        b_size,channels,h,w = real_cpu.shape

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        patch = 128
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_real = F.crop(real_cpu, top, left, patch, patch)
        outputOrig = origD(cropped_real).view(-1)

        patch = 16
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_real = F.crop(real_cpu, top, left, patch, patch)
        outputMini = miniD(cropped_real).view(-1)

        # Calculate loss on all-real batch
        errD_real_orig = criterion(outputOrig, label)
        # Calculate gradients for D in backward pass
        errD_real_orig.backward()

        errD_real_mini = criterion(outputMini, label)
        errD_real_mini.backward()

        D_x = outputOrig.mean().item() + outputMini.mean().item()

        ## Train with all-fake batch

        # Generate fake image batch with G
        fake = netG(simdata)
        label.fill_(fake_label)

        # Classify all fake batch with D
        patch = 128
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake = F.crop(fake.detach(), top, left, patch, patch)
        outputOrig = origD(cropped_fake).view(-1)

        patch = 16
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake = F.crop(fake.detach(), top, left, patch, patch)
        outputMini = miniD(cropped_fake).view(-1)


        # Calculate D's loss on the all-fake batch
        errD_fake_orig = criterion(outputOrig, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake_orig.backward()

        errD_fake_mini = criterion(outputMini, label)
        errD_fake_mini.backward()


        D_G_z1 = outputOrig.mean().item() + outputMini.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = (errD_real_orig+errD_real_mini) + (errD_fake_orig + errD_fake_mini)
        # Update D
        optimizerDorig.step()
        optimizerDmini.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        patch = 128
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake_orig = F.crop(fake, top, left, patch, patch)
        outputOrig = origD(cropped_fake_orig).view(-1)

        patch = 16
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake_mini = F.crop(fake, top, left, patch, patch)
        outputMini = miniD(cropped_fake_mini).view(-1)

        # Calculate G's loss based on this output
        errG = criterion(outputOrig, label) + criterion(outputMini, label)

        temp = outputOrig
        # Calculate gradients for G
        errG.backward()

        D_G_z2 = outputOrig.mean().item() + outputMini.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print(time.time()-start)
            start = time.time()

        if (iters%50 ==0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(simdata).detach().cpu().numpy()

            # display patches with large error
            for x in range(len(temp)):
                if(temp[x]<0.5):
                    patches = ((cropped_fake_orig[x][0].detach().cpu().numpy()*0.5)+0.5)*255.
                    patches = Image.fromarray(patches.astype(np.uint8),'L')
                    patches.save('patches/patch'+str(x) + '_' + str(epoch)+'_'+str(i)+'.png')

            img = ((fake[0][0]*0.5)+0.5)*255.
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('fake'+str(epoch)+'_'+str(i)+'.png')
            print(simpath[0])
            temp = (simdata[0][0]).detach().cpu().numpy()
            temp = ((temp*0.5)+0.5)*255.
            temp = Image.fromarray(temp.astype(np.uint8),'L')
            temp.save('orig'+str(epoch)+'_'+str(i)+'.png')

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

print('evaluating')
for simdata in enumerate(simtest):
    j, simdata = simdata
    simdata, simpath = simdata
    simdata = simdata.to(device)
    for i in range(batch_size):
        if '1-300135-15' in simpath[i]:
            with torch.no_grad():
                fake = netG(simdata).detach().cpu().numpy()
                fake = ((fake[i][0]*0.5)+0.5)*255.
                fake = Image.fromarray(fake.astype(np.uint8),'L')
                fake.save('fake.png')
                print(simpath[i])
                temp = (simdata[i][0]).detach().cpu().numpy()
                temp = ((temp*0.5)+0.5)*255.
                temp = Image.fromarray(temp.astype(np.uint8),'L')
                temp.save('orig.png')

# distribution
occur = dict()
for simdata in enumerate(simtest):
    j, simdata = simdata
    simdata, simpath = simdata
    simdata = simdata.to(device)
    with torch.no_grad():
        fake = netG(simdata).detach().cpu().numpy().flatten()
        fake = ((fake*0.5)+0.5)*255.
    unique, counts = np.unique(fake.astype(np.uint8), return_counts=True)
    temp = dict(zip(unique, counts))
    occur = merge_two_dicts(occur, temp)
print(occur)

# loss graph
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('lossgraph.png')
