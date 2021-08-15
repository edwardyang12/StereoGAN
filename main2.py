import torch
import torchvision.transforms.functional as F
import itertools
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

from nets.discriminator import OrigDiscriminator as Discriminator
from nets.generator import ResGenerator as Generator
from nets.utils import ReplayBuffer, weights_init
from datasets.custom_test import CustomDatasetTest
from datasets.custom_dataset import CustomDataset

lr = 0.0002
num_epochs = 15
batch_size = 7
beta1 = 0.5
num_workers = 0
ngpu = 1
patch = 128 # patch size
size = 512 # picture size

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

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG_A2B = nn.DataParallel(netG_A2B, list(range(ngpu)))
    netG_B2A = nn.DataParallel(netG_B2A, list(range(ngpu)))

netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD_A = nn.DataParallel(netD_A, list(range(ngpu)))
    netD_B = nn.DataParallel(netD_B, list(range(ngpu)))

netD_A.apply(weights_init)
netD_B.apply(weights_init)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batch_size, 1, size, size)
input_B = Tensor(batch_size, 1, size, size)
target_real = Variable(Tensor(batch_size).fill_(real_label), requires_grad=False)
target_fake = Variable(Tensor(batch_size).fill_(fake_label), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

G_losses = []
G_identity_losses = []
G_GAN_losses = []
G_cycle_losses = []
D_losses = []
iters = 0

###### Training ######
print("Starting Training Loop...")
for epoch in range(num_epochs):
    start = time.time()
    for data, simdata in zip(enumerate(dataloader), enumerate(simloader)):
        i, data = data
        data, path = data
        j, simdata = simdata
        simdata, simpath = simdata

        b_size,channels,h,w = data.shape

        real_A = Variable(input_A.copy_(data))
        real_B = Variable(input_B.copy_(simdata))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_same_B = F.crop(same_B, top, left, patch, patch)
        cropped_real_B = F.crop(real_B, top, left, patch, patch)
        loss_identity_B = criterion_identity(cropped_same_B, cropped_real_B)*5.0

        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_same_A = F.crop(same_A, top, left, patch, patch)
        cropped_real_A = F.crop(real_A, top, left, patch, patch)
        loss_identity_A = criterion_identity(cropped_same_A, cropped_real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake = F.crop(fake_B, top, left, patch, patch)
        pred_fake = netD_B(cropped_fake).view(-1)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake = F.crop(fake_A, top, left, patch, patch)
        pred_fake = netD_A(cropped_fake).view(-1)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_recovered_A = F.crop(recovered_A, top, left, patch, patch)
        cropped_real_A = F.crop(real_A, top, left, patch, patch)
        loss_cycle_ABA = criterion_cycle(cropped_recovered_A, cropped_real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_recovered_B = F.crop(recovered_B, top, left, patch, patch)
        cropped_real_B = F.crop(real_B, top, left, patch, patch)
        loss_cycle_BAB = criterion_cycle(cropped_recovered_B, cropped_real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_real = F.crop(real_A, top, left, patch, patch)
        pred_real = netD_A(cropped_real).view(-1)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake = F.crop(fake_A.detach(), top, left, patch, patch)
        pred_fake = netD_A(cropped_fake).view(-1)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_real = F.crop(real_B, top, left, patch, patch)
        pred_real = netD_B(cropped_real).view(-1)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        top = np.random.randint(0,h-patch)
        left = np.random.randint(0,w-patch)
        cropped_fake = F.crop(fake_B.detach(), top, left, patch, patch)
        pred_fake = netD_B(cropped_fake).view(-1)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()


        if i % 25 == 0:
            print("====== ", i, len(dataloader), epoch)
            print('loss_G: '+ str(loss_G.item()) + ' loss_G_identity: ' + str((loss_identity_A + loss_identity_B).item()) +  ' loss_G_GAN: ' + str((loss_GAN_A2B + loss_GAN_B2A).item()) + ' loss_G_cycle: ' +  str((loss_cycle_ABA + loss_cycle_BAB).item()))
            print('loss_D: ' + str((loss_D_A + loss_D_B).item()))
            print(time.time()-start)
            start = time.time()

        if (iters%25 ==0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_B = netG_B2A(real_B).detach().cpu().numpy()
                fake_A = netG_A2B(real_A).detach().cpu().numpy()

            img = ((data[0][0]*0.5)+0.5)*255.
            img = img.detach().cpu().numpy()
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('real'+str(epoch)+'_'+str(i)+'.png')

            img = ((simdata[0][0]*0.5)+0.5)*255.
            img = img.detach().cpu().numpy()
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('sim'+str(epoch)+'_'+str(i)+'.png')

            img = ((fake_A[0][0]*0.5)+0.5)*255.
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('fakeReal'+str(epoch)+'_'+str(i)+'.png')

            img = ((fake_B[0][0]*0.5)+0.5)*255.
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('fakeSim'+str(epoch)+'_'+str(i)+'.png')

            print(simpath[0], path[0])

        G_losses.append(loss_G.item())
        D_losses.append((loss_D_A + loss_D_B).item())
        G_GAN_losses.append((loss_GAN_A2B + loss_GAN_B2A).item())
        iters += 1

print('evaluating')
for simdata in enumerate(simtest):
    j, simdata = simdata
    simdata, simpath = simdata
    simdata = simdata.to(device)
    for i in range(batch_size):
        if '1-300135-15' in simpath[i]:
            with torch.no_grad():
                fake_B = netG_B2A(simdata).detach().cpu().numpy()

            img = ((simdata[0][0]*0.5)+0.5)*255.
            img = img.detach().cpu().numpy()
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('sim.png')

            img = ((fake_B[0][0]*0.5)+0.5)*255.
            img = Image.fromarray(img.astype(np.uint8),'L')
            img.save('fakeSim.png')
            break


# loss graph
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(G_GAN_losses,label="G_GAN")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('lossgraph.png')
