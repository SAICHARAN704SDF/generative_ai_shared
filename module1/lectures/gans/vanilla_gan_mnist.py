"""
Vanilla (Fully-Connected) GAN on MNIST
----------------------------------------
A minimal, teaching-focused implementation of the original GAN
architecture (Goodfellow et al., 2014) using simple MLPs for both
the Generator and Discriminator.

Run: python vanilla_gan_mnist.py
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
IMG_DIM = 28 * 28          # flattened MNIST image
BATCH_SIZE = 128
EPOCHS = 50
LR = 2e-4
SAMPLE_DIR = "gan_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)


# ----------------------------
# Data
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # scale to [-1, 1] to match Tanh output
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------
# Generator: noise -> image
# ----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_dim=IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()          # output range [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# ----------------------------
# Discriminator: image -> real/fake probability
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, img_dim=IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()        # probability of being real
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Setup
# ----------------------------
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# We reuse the SAME BCELoss for both networks. What changes between them
# is not the loss function itself, but which LABELS we feed it:
#   BCE(p, y) = -[ y*log(p) + (1-y)*log(1-p) ]
# where p = D's predicted probability (sigmoid output), y = target label.
#   - For D: y is the TRUE label (1 for real, 0 for fake) -> standard classifier training.
#   - For G: y is a "wish" label (always 1) -> G is scored as if its fakes were real,
#            and the resulting gradient pushes G to make that wish come true.
# See the training loop below for exactly where each of these is used.
criterion = nn.BCELoss()

# Separate optimizers -> each one only ever touches its own network's
# parameters (G.parameters() vs D.parameters()), which is what actually
# guarantees "only D updates" or "only G updates" in each step below.
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

g_losses, d_losses = [], []

# Fixed noise vector to visualize generator progress consistently across epochs
# We use these fixed noise samples to generate images at the end of each epoch, 
# so we can see how the generator improves over time.
fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE)

# plot a sample grid of generated images at the end of each epoch
def save_sample_grid(epoch):
    G.eval()
    with torch.no_grad():
        fake_imgs = G(fixed_noise).view(-1, 28, 28).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_imgs[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{SAMPLE_DIR}/epoch_{epoch:03d}.png")
    plt.close()
    G.train()


# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(1, EPOCHS + 1):
    epoch_g_loss, epoch_d_loss = 0.0, 0.0

    for real_imgs, _ in train_loader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size, -1).to(DEVICE)

        # These are the TRUE labels D is trained against: real images are
        # actually real (1), fake images are actually fake (0).
        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # ============================================================
        # Step A: Train the Discriminator (a plain binary classifier)
        # Goal: D should output ~1 for real images, ~0 for fake images.
        # G's weights are NOT updated in this step.
        # ============================================================
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        # .detach() cuts fake_imgs off from G's computation graph. We don't
        # need gradients flowing into G here since only opt_D.step() runs
        # below -- detaching just avoids building/backpropping through
        # G's graph unnecessarily (a minor efficiency/clarity choice).
        fake_imgs = G(z).detach()

        # D(real_imgs) should be close to 1 -> compared against real_labels=1.
        # Small loss if D correctly says "real"; large loss if D is fooled.
        d_real_loss = criterion(D(real_imgs), real_labels)

        # D(fake_imgs) should be close to 0 -> compared against fake_labels=0.
        # Small loss if D correctly says "fake"; large loss if D is fooled.
        d_fake_loss = criterion(D(fake_imgs), fake_labels)

        d_loss = d_real_loss + d_fake_loss   # total classification loss

        opt_D.zero_grad()
        d_loss.backward()   # gradients computed only w.r.t. D's parameters
        opt_D.step()        # only D's weights are updated (opt_D only knows about D)

        # ============================================================
        # Step B: Train the Generator (D acts only as a frozen judge)
        # Goal: fool D into outputting ~1 (i.e. "real") for these fakes.
        # D's weights are NOT updated in this step (opt_D.step() is never
        # called here) -- but gradients DO flow *through* D's layers on
        # the way back to G, since D is part of the computation graph
        # connecting G's output to g_loss.
        # ============================================================
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z)   # NOT detached this time -- we need the graph
                            # to reach back into G's parameters.

        # This is the key line that confuses people: we score the FAKE
        # images against real_labels (1), not fake_labels (0). We are
        # asking "how far is D's output from saying these are real?"
        #   - If D is already fooled (D(fake) ~ 1): loss is small,
        #     weak gradient -- G doesn't need to change much.
        #   - If D isn't fooled (D(fake) ~ 0): loss is large,
        #     strong gradient -- G is pushed hard to change its output
        #     so that D is more likely to be fooled next time.
        # This is the "non-saturating" formulation: mathematically
        # equivalent to maximizing log(D(G(z))), which gives much
        # stronger gradients early in training than the original
        # minimize-log(1-D(G(z))) formulation (which saturates when
        # D easily rejects poor early fakes).
        g_loss = criterion(D(fake_imgs), real_labels)

        opt_G.zero_grad()
        g_loss.backward()   # gradients flow: g_loss -> D's layers (used, not updated) -> G's layers
        opt_G.step()        # only G's weights are updated (opt_G only knows about G)

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
plt.title("Vanilla GAN Training Losses")
plt.legend()
plt.savefig(f"{SAMPLE_DIR}/loss_curve.png")
plt.show()

print(f"\nTraining complete. Sample grids and loss curve saved in '{SAMPLE_DIR}/'")
