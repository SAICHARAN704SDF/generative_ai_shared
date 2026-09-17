"""
DCGAN (Deep Convolutional GAN) on MNIST
----------------------------------------
A teaching-focused DCGAN implementation, following the architectural
guidelines from Radford, Metz & Chintala (2015).

Compare this file directly against vanilla_gan_mnist.py: the TRAINING
LOOP (Step A / Step B, BCELoss usage, non-saturating trick) is IDENTICAL.
Only the network architectures (Generator, Discriminator) have changed
from MLPs to CNNs. This is the single most important thing to notice.

MNIST images are resized from 28x28 to 32x32 so that a clean
4 -> 8 -> 16 -> 32 upsampling path (doubling each stage) can be used.

Run: python dcgan_mnist.py
Requires: torch, torchvision, matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
IMG_SIZE = 32           # resized from MNIST's native 28x28
CHANNELS = 1            # MNIST is grayscale
BATCH_SIZE = 128
EPOCHS = 30
LR = 2e-4               # standard DCGAN learning rate
BETA1 = 0.5             # DCGAN paper recommends beta1=0.5 for Adam (default is 0.9)
SAMPLE_DIR = "dcgan_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)


# ----------------------------
# Data
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),                # 28x28 -> 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))         # scale to [-1, 1] to match Tanh output
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------
# Weight initialization (DCGAN paper recommendation)
# Conv/ConvTranspose weights ~ N(0, 0.02); BatchNorm weights ~ N(1, 0.02)
# ----------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ----------------------------
# Generator: noise -> image, via transposed convolutions
# Pattern: spatial size DOUBLES, channel depth HALVES, at each stage.
#   z (100,1,1) -> 4x4x256 -> 8x8x128 -> 16x16x64 -> 32x32x1
# ----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, channels=CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            # "Project & reshape" step: treat the latent vector as a
            # 1x1 spatial input, and use a transposed conv to expand it
            # into a small 4x4 spatial seed. This is the one place a
            # DCGAN generator effectively starts from something FC-like.
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 16x16x64 -> 32x32xCHANNELS  (final image, no BatchNorm at output -- guideline)
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()          # output range [-1, 1], matches normalized real images
        )

    def forward(self, z):
        # z arrives as (batch, latent_dim) -- reshape to (batch, latent_dim, 1, 1)
        # so it can be treated as a 1x1 spatial "image" for the first ConvTranspose2d
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z)


# ----------------------------
# Discriminator: image -> real/fake probability, via strided convolutions
# Pattern: spatial size HALVES, channel depth DOUBLES, at each stage
# (mirror image of the Generator).
#   32x32x1 -> 16x16x64 -> 8x8x128 -> 4x4x256 -> single output
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            # 32x32xCHANNELS -> 16x16x64  (no BatchNorm at input layer -- guideline)
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4x256 -> 1x1x1 (collapse remaining spatial extent -> single score)
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()        # probability of being real
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, 1)   # flatten (batch,1,1,1) -> (batch,1) to match BCELoss expectations


# ----------------------------
# Setup
# ----------------------------
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
G.apply(weights_init)
D.apply(weights_init)

# Exactly the same loss/label mechanism as the vanilla GAN:
#   - D is trained with TRUE labels (1=real, 0=fake) -> standard classifier training
#   - G is trained with a "wish" label (always 1) -> scored as if its fakes were real
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

g_losses, d_losses = [], []

# Fixed noise vector to visualize generator progress consistently across epochs
fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE)


def save_sample_grid(epoch):
    G.eval()
    with torch.no_grad():
        fake_imgs = G(fixed_noise).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_imgs[i].squeeze(0), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{SAMPLE_DIR}/epoch_{epoch:03d}.png")
    plt.close()
    G.train()


# ----------------------------
# Training Loop
# (Identical in structure to the vanilla GAN's training loop -- only the
#  network architectures above have changed. If this looks familiar,
#  that's the point: DCGAN is the same adversarial training algorithm,
#  applied to a better architecture.)
# ----------------------------
for epoch in range(1, EPOCHS + 1):
    epoch_g_loss, epoch_d_loss = 0.0, 0.0

    for real_imgs, _ in train_loader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(DEVICE)   # kept as (B, C, H, W) -- NOT flattened, unlike vanilla GAN

        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # ---- Step A: Train Discriminator (G frozen) ----
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z).detach()

        d_real_loss = criterion(D(real_imgs), real_labels)
        d_fake_loss = criterion(D(fake_imgs), fake_labels)
        d_loss = d_real_loss + d_fake_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ---- Step B: Train Generator (D frozen, used only as a judge) ----
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z)
        # Non-saturating trick: score fakes against real_labels=1 (see vanilla GAN notes)
        g_loss = criterion(D(fake_imgs), real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()

    avg_d = epoch_d_loss / len(train_loader)
    avg_g = epoch_g_loss / len(train_loader)
    d_losses.append(avg_d)
    g_losses.append(avg_g)

    print(f"Epoch [{epoch}/{EPOCHS}]  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")

    if epoch % 5 == 0 or epoch == 1:
        save_sample_grid(epoch)

# ----------------------------
# Plot loss curves
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DCGAN Training Losses")
plt.legend()
plt.savefig(f"{SAMPLE_DIR}/loss_curve.png")
plt.show()

print(f"\nTraining complete. Sample grids and loss curve saved in '{SAMPLE_DIR}/'")
