import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import dataloader
import torchvision.utils as vutils
import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = True
device = torch.device("cuda:0")
nz = 100
ngf = 64
ndf = 64
ngpu = 1
nc = 1
lr = 0.0002
image_size = (50, 40)
batchSize = 64
beta1 = 0.5
num_epochs = 25
netG = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Models'
netD = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Models'
outf = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Output/'

dataroot = 'C:/Users/User/Desktop/PhD_Files/Input/Bosphorus/GAN/'
dataset = dset.ImageFolder(root=dataroot,
							transform=transforms.Compose([
							transforms.Grayscale(num_output_channels=1),
							transforms.Resize(image_size),
							transforms.ToTensor()
							]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)


# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		torch.nn.init.normal_(m.weight, 1.0, 0.02)
		torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(     nz, ngf * 8, (3, 2), 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
			)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
		return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)


class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(                             					# 1 x 50 x 40
			nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),				   					# 64 x 32 x 32
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),									# 128 x 16 x 16
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),									# 256 x 8 x 8
			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),									# 512 x 4 x 4
			nn.Conv2d(512, 1, kernel_size=(3, 2), stride=1, padding=0, bias=False),
			nn.Sigmoid()
			)

	def forward(self, input):
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)

		return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
	# For each batch in the dataloader
	for i, data in enumerate(dataloader, 0):

		############################
		# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		###########################
		## Train with all-real batch
		netD.zero_grad()
		# Format batch
		real_cpu = data[0].to(device)
		b_size = real_cpu.size(0)
		label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
		# Forward pass real batch through D
		output = netD(real_cpu).view(-1)

		# Calculate loss on all-real batch
		errD_real = criterion(output, label)
		# Calculate gradients for D in backward pass
		errD_real.backward()
		D_x = output.mean().item()

		## Train with all-fake batch
		# Generate batch of latent vectors
		noise = torch.randn(b_size, nz, 1, 1, device=device)
		# Generate fake image batch with G
		fake = netG(noise)
		label.fill_(fake_label)
		# Classify all fake batch with D
		output = netD(fake.detach()).view(-1)

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
		output = netD(fake).view(-1)
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

		# Save Losses for plotting later
		G_losses.append(errG.item())
		D_losses.append(errD.item())

		# Check how the generator is doing by saving G's output on fixed_noise
		if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
			with torch.no_grad():
				fake = netG(fixed_noise).detach().cpu()
				img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
				for image in fake.clone().numpy():
					plt.imsave(outf + str(random.randint(1, 77777)) + '.png', image[0], cmap=plt.get_cmap('gray'))

		iters += 1

# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(G_losses,label="G")
# plt.plot(D_losses,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
