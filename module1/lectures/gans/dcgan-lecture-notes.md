# Deep Convolutional GANs (DCGAN) — Lecture Notes

## 1. Why DCGAN? (Motivation from Vanilla GAN)

The vanilla GAN used fully connected (MLP) layers for both G and D. Two structural weaknesses:

1. **No spatial locality awareness** — flattening a 28×28 image into a 784-dim vector throws away the notion that neighboring pixels are related. The network has to learn spatial structure from scratch.
2. **Parameter explosion at scale** — fully connected layers between image-sized vectors don't scale to larger, real-world image sizes.

**DCGAN (Radford, Metz & Chintala, 2015)** replaces MLPs with **CNNs** — the same tool that dominates image classification, adapted for generation. It was one of the first papers to show GANs producing genuinely convincing images, and it introduced architectural guidelines that became a near-default recipe for GAN design.

---

## 2. The Core Architectural Shift

| | Vanilla GAN | DCGAN |
|---|---|---|
| Generator | MLP: noise → flat vector | CNN: noise → small spatial seed → progressively **upsampled** via transposed convolutions |
| Discriminator | MLP: flat vector → probability | CNN: image → progressively **downsampled** via strided convolutions → probability |

**Symmetry to emphasize:** D shrinks spatial size while increasing channel depth as it goes deeper (like a normal CNN classifier). G does the exact reverse — expands spatial size while decreasing channel depth.

---

## 3. Transposed Convolution (the trickiest concept — spend time here)

- A regular **strided convolution** downsamples: e.g. 28×28 → 14×14.
- A **transposed convolution** upsamples: e.g. 7×7 → 14×14. It spreads each input value across a larger output region using learned weights, via a similar sliding-window mechanism.

**Important nuance:** transposed convolution is **not** the mathematical inverse of convolution — it doesn't "undo" a convolution. It's a *learned upsampling operation*, trained via backprop like any other layer.

**Framing for students:**
> Convolution asks: *"what's the summary of this neighborhood?"*
> Transposed convolution asks: *"given this value, how should it spread out into a neighborhood?"*

---

## 4. DCGAN Architectural Guidelines (the practical checklist)

| Guideline | What it means | Why it matters |
|---|---|---|
| Replace pooling with strided convolutions | Use stride-2 convs in D instead of max/avg pooling | Pooling discards info in a fixed way; strided convs let the network *learn* the best downsampling |
| Use transposed convolutions in G | For upsampling noise → image | Lets G learn to "grow" spatial detail rather than relying on naive resizing |
| Use BatchNorm in G and D | Except G's output layer and D's input layer | Stabilizes training through deep conv stacks; excluding those layers avoids distorting real/generated pixel values directly |
| Remove fully connected hidden layers | Except an initial "project & reshape" in G | Keeps spatial structure intact throughout; avoids FC parameter blowup |
| ReLU in G (Tanh at output) | Standard ReLU in hidden layers | Empirically helps G learn faster; Tanh matches [-1,1] normalized pixel range |
| LeakyReLU in D (all layers) | Small negative slope instead of hard zero | Prevents "dead neurons" — keeps gradient signal flowing through D, which matters since D's gradient quality directly affects G's learning |

---

## 5. Generator Architecture — Worked Example (64×64 output)

```
z (100-dim noise)
  → project & reshape → 4×4×1024
  → transposed conv (stride 2) → 8×8×512    + BatchNorm + ReLU
  → transposed conv (stride 2) → 16×16×256  + BatchNorm + ReLU
  → transposed conv (stride 2) → 32×32×128  + BatchNorm + ReLU
  → transposed conv (stride 2) → 64×64×3    + Tanh   (final image)
```

**Pattern:** spatial size **doubles**, channel depth **halves** at each stage.

## 6. Discriminator Architecture — Mirror Image

```
image (64×64×3)
  → conv (stride 2) → 32×32×128   + LeakyReLU
  → conv (stride 2) → 16×16×256   + BatchNorm + LeakyReLU
  → conv (stride 2) → 8×8×512     + BatchNorm + LeakyReLU
  → conv (stride 2) → 4×4×1024    + BatchNorm + LeakyReLU
  → flatten → single output + Sigmoid
```

**Pattern:** spatial size **halves**, channel depth **doubles** at each stage — the reverse of G.

---

## 7. Why DCGAN Trains More Stably (Connects Back to Convergence Issues)

- **BatchNorm** reduces internal covariate shift across the deep conv stack, helping avoid the G/D imbalance that leads to vanishing gradients.
- **LeakyReLU in D** keeps gradient signal alive where standard ReLU would zero it out — critical, since a "dead" D gives G no useful learning signal at all.
- **Convolutional weight sharing** acts as a structural prior, making it harder for the network to simply memorize a few outputs — this tends to reduce (not eliminate) mode collapse compared to MLPs.

**Honesty point for class:** DCGAN's guidelines *reduce* instability — they don't *solve* it. Mode collapse and oscillation are still possible. DCGAN was a major empirical improvement, not a theoretical fix to the minimax game's fundamental instability.

---

## 8. What Actually Changes vs. the Vanilla GAN Code

**The training loop stays exactly the same** — Step A (train D with real/fake labels), Step B (train G with the non-saturating trick), BCELoss usage — none of that changes. Only the **network architectures** change:

- `Generator`: `nn.Linear` stack → `nn.ConvTranspose2d` stack + `BatchNorm2d` + `ReLU`, ending in `Tanh`
- `Discriminator`: `nn.Linear` stack → `nn.Conv2d` stack + `BatchNorm2d` + `LeakyReLU`, ending in `Sigmoid`
- Data: images stay as `(C, H, W)` tensors instead of being flattened to vectors

**Key teaching point:** DCGAN is not a different training algorithm — it's a better architecture plugged into the same adversarial training framework from the vanilla GAN lecture.

---

## 9. Summary Table

| Concept | One-line takeaway |
|---|---|
| Motivation | MLPs ignore spatial structure and don't scale to larger images |
| Transposed convolution | Learned upsampling, not the mathematical inverse of convolution |
| G architecture | Noise → small spatial seed → upsample (double size, halve channels) each stage |
| D architecture | Image → downsample (halve size, double channels) each stage → probability |
| BatchNorm | Stabilizes deep conv training; excluded at G's output / D's input |
| LeakyReLU in D | Prevents dead gradients, preserves signal quality for G |
| Training loop | Identical to vanilla GAN — only architecture changed |
| Stability | Improved empirically, not theoretically guaranteed |

---

## 10. Lab Exercise: DCGAN on MNIST

See accompanying `dcgan_mnist.py` — trains a convolutional GAN on MNIST digits (resized to 32×32 for a clean 4→8→16→32 upsampling path).

**What to observe while running it:**
1. Compare sample quality and sharpness against the vanilla GAN's outputs from the previous lab, at a similar epoch count
2. Watch how much faster recognizable digit shapes emerge compared to the MLP version
3. Plot loss curves — are they more stable than the vanilla GAN's?
4. Try removing BatchNorm from the Generator — what happens to training stability and output quality?

**Discussion questions for students:**
- Why does removing BatchNorm tend to destabilize training more in deeper conv networks than in shallow MLPs?
- If you wanted to generate 128×128 images instead of 32×32, how many additional upsampling stages would you need, and how would channel depth change?
- Why is LeakyReLU used in D but plain ReLU in G? What would happen if you swapped them?
