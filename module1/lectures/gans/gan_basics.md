# Generative Adversarial Networks (GANs) — Lecture Notes

## Before We Begin: Game Theory and Nash Equilibrium

**Game theory** studies situations where each player's outcome depends on what the other players do. In a GAN, the two players are neural networks:

- The **generator (G)** acts like an art forger: it creates fake samples and tries to make them look real.
- The **discriminator (D)** acts like an art detective: it tries to distinguish real samples from generated ones.

Each network's performance depends on the other. A better detective gives the forger a harder challenge, and a better forger gives the detective a harder challenge. The original GAN's **minimax objective** expresses this competition mathematically: D maximizes a score for distinguishing real from fake, while G minimizes that same score by fooling D. This is a **zero-sum game**—one player's gain in that score is the other's loss.

### What Is Nash Equilibrium?

A **Nash equilibrium** is a situation where **neither player can improve its own outcome by changing its strategy alone, while the other player's strategy stays fixed**. In a GAN, a network's strategy is the behavior determined by its weights.

The ideal GAN equilibrium looks like this:

1. **G matches the real data distribution**, including its variety—not just a few convincing examples.
2. **D cannot distinguish real from generated samples.** With equally likely real and fake inputs, its best prediction is `0.5` for samples from that distribution.
3. **Neither can improve alone:** D has no distinguishing clues to learn, and with D fixed at `0.5`, changing G cannot improve G's score.

Think of a forger whose work has exactly the same statistical patterns as genuine art: even the best detective has no useful clue about its origin.

### How Does Training Try to Reach It?

Training aims to approach this equilibrium through repeated, alternating updates:

1. **Hold G fixed and train D** to better distinguish real samples from G's current fakes.
2. **Hold D's weights fixed and train G**, using feedback through D to make its fakes more convincing.
3. **Repeat**, so each network responds to the other's changing behavior.

In the ideal theoretical setting, with enough model capacity, an optimal discriminator at each stage, and suitable generator updates, this process can approach the real data distribution. **Ordinary neural-network training does not guarantee reaching a Nash equilibrium**: the networks can oscillate, learn unevenly, or generate too little variety.

In practice, balancing learning rates and update frequencies, and using a generator loss that gives useful feedback (Section 5), can help training progress. Monitor both sample quality and variety. **A discriminator output near `0.5` alone does not prove equilibrium**—a poorly trained discriminator can also be uncertain.

---

## 1. What is Generative Modeling?

- **Discriminative models** learn `P(y|x)` — given an input, predict a label. (e.g. "is this a cat or dog?")
- **Generative models** learn (implicitly or explicitly) `P(x)` — the underlying data distribution — so they can **produce new samples** that look like they came from that distribution.

GANs are a generative model that learns to generate data **without explicitly modeling the probability density function** — instead, it learns through a competitive game.

---

## 2. The Core Idea: A Game Between Two Networks

**Analogy:** A counterfeiter (Generator) is trying to produce fake currency good enough to fool a detective (Discriminator). The detective is trying to catch every fake. As the detective gets better at spotting fakes, the counterfeiter is forced to improve. As the counterfeiter improves, the detective must sharpen further. Both improve *because of* the other's pressure.

| Network | Role | Input | Output |
|---|---|---|---|
| **Generator (G)** | Produces fake data | Random noise vector `z` | Fake sample `G(z)` |
| **Discriminator (D)** | Distinguishes real vs fake | A sample (real or fake) | Probability it's real (0–1) |

- `z` is sampled from a simple known distribution (e.g. Gaussian or Uniform) — often called the **latent space**.
- G's job: map noise → realistic data.
- D's job: output high probability for real data, low probability for fake data.

---

## 3. The Minimax Objective

$$\min_G \max_D \; V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

Reading it in plain terms:
- **D wants to maximize** this — correctly classify real data as real (`D(x) → 1`) and fake data as fake (`D(G(z)) → 0`).
- **G wants to minimize** this — make D classify its fakes as real (`D(G(z)) → 1`), which pulls the second term down.

**Theoretical equilibrium:** when G perfectly matches the real data distribution, D can't tell real from fake at all — `D(x) = 0.5` everywhere. This is the Nash equilibrium of the game, at least in theory.

---

## 4. Adversarial Training Loop

Training alternates between two steps, repeated every iteration:

**Step A — Train the Discriminator (G is frozen)**
1. Sample a batch of real data `x` from the dataset
2. Sample a batch of noise `z`, generate fakes: `x_fake = G(z)`
3. Compute D's loss on real (label = 1) and fake (label = 0) samples
4. Backpropagate and update **only D's weights**

**Step B — Train the Generator (D is frozen)**
1. Sample a new batch of noise `z`, generate fakes: `x_fake = G(z)`
2. Pass fakes through D, but use label = 1 (i.e. G "wants" D to say these are real)
3. Backpropagate and update **only G's weights**

Repeat A → B → A → B ... for many epochs.

> **Common confusion to clarify in class:** during Step B, gradients flow *through* D (to compute how G's output should change to fool D better), but D's own weights are not updated in that step. D acts purely as a critic here.

---

## 5. The Non-Saturating Loss Trick

In practice, G is **not** trained with the original `log(1 - D(G(z)))` term. Early in training, D easily rejects G's poor fakes, so `D(G(z)) ≈ 0`, and this term saturates — its gradient becomes almost flat, so G barely learns.

**Fix:** train G to instead **maximize** `log(D(G(z)))` directly (equivalently, minimize `-log(D(G(z)))`). This gives much stronger gradients early in training, when G most needs them. This is what's used in almost all practical implementations, including the one below.

---

## 6. Convergence Issues (Important — exam-worthy)

Unlike standard supervised learning (one loss going downhill), GAN training is a **two-player game** — there's no guarantee of stable convergence. Key failure modes:

**a) Mode Collapse**
- G discovers a small set of outputs (sometimes just one) that reliably fool D, and keeps producing variations of just that — losing diversity.
- Example: a GAN trained on MNIST digits that only ever generates "1"s, regardless of the input noise.
- Why it happens: G is optimizing to fool the *current* D, not to match the full data distribution — it can "hide" in a blind spot.

**b) Vanishing Gradients**
- If D becomes too strong too early (near-perfect at spotting fakes), the gradient signal back to G becomes very small — G stops improving.
- The non-saturating loss trick (Section 5) partially addresses this, but the underlying tension (D outpacing G) remains a core training challenge.

**c) Oscillation / Non-Convergence**
- Since G and D are chasing a moving target (each other), losses can oscillate indefinitely instead of settling into equilibrium.
- There's no built-in stopping criterion like "loss reached zero" — you monitor sample quality visually alongside the loss curves.

**Practical mitigations (worth mentioning, deeper coverage not needed at this stage):** balancing G/D learning rates, label smoothing, adding noise to D's inputs, architectural fixes (covered later in DCGAN).

---

## 7. Vanilla / Basic GAN — Architecture

The original (Goodfellow et al., 2014) formulation uses simple **fully connected (MLP)** networks for both G and D — no convolutions, no spatial awareness of image structure.

**Generator**
- Input: noise vector `z` (e.g. 100-dim)
- Few fully connected layers with increasing width
- Output: flattened image vector (e.g. 784-dim for 28×28 MNIST), squashed with `Tanh` to range [-1, 1]

**Discriminator**
- Input: flattened image vector (784-dim)
- Few fully connected layers with decreasing width
- Output: single value in [0,1] via `Sigmoid` — probability the input is real

**Limitations (motivates DCGAN, covered next lecture):**
- No spatial/local structure awareness — treats an image as a flat vector, ignoring pixel neighborhoods
- Produces blurry, low-fidelity images
- Scales poorly to larger or more complex images

---

## 8. Summary Table

| Concept | One-line takeaway |
|---|---|
| Generator | Learns noise → data mapping |
| Discriminator | Learns real vs fake classification |
| Minimax loss | G minimizes, D maximizes the same value function |
| Training loop | Alternate: update D, then update G, repeat |
| Non-saturating trick | G maximizes `log D(G(z))` instead, for stronger early gradients |
| Mode collapse | G produces limited variety of outputs |
| Vanishing gradients | D too strong → G gets no learning signal |
| Vanilla GAN | MLP-based G and D; simple but blurry/low-quality outputs |

---

## 9. Lab Exercise: Vanilla GAN on MNIST

See accompanying `vanilla_gan_mnist.py` — trains a simple fully-connected GAN on MNIST digits.

**What to observe while running it:**
1. Plot G and D losses per epoch — do they stabilize, oscillate, or does one collapse to near-zero?
2. Visualize generated samples every few epochs — note how blurry/noisy early outputs are vs later ones
3. Try training for many more epochs than needed — do you see any signs of mode collapse (limited variety of digits)?
4. Try deliberately unbalancing training (e.g. update D 5x per G update) — what happens to sample quality?

**Discussion questions for students:**
- Why does D's loss often decrease faster than G's early in training? What does that imply?
- If D's accuracy on real-vs-fake reaches ~100% and stays there, what training problem does this suggest?
- How would you modify this architecture to work on larger, colored images? (Bridges directly into DCGAN.)
