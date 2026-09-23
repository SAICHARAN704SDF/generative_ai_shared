# Recurrent Neural Networks (RNN) — The Basics

A tutorial covering the core mechanics of a vanilla RNN: notation, the forward pass, the RNN cell, and the loss function.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $T_x$ | Length of the input sequence |
| $T_y$ | Length of the output sequence |
| $x^{\langle i \rangle \langle t \rangle}$ | Input at time step $t$, for training example $i$ |
| $y^{\langle i \rangle \langle t \rangle}$ | True target at time step $t$, for training example $i$ |
| $\hat{y}^{\langle t \rangle}$ | Model prediction at time step $t$ |
| $n_x$ | Dimension of the input $x$ |
| $n_h$ | Dimension of the hidden layer (hidden state) |
| $n_y$ | Dimension of the output $y$ |
| $a^{\langle t \rangle}$ | Hidden (activation) state at time step $t$ |
| $a_0$ | Initial hidden state — usually a vector of zeros |

### Weight matrices

For one training example, the vectors entering or leaving an RNN cell are:

- $a^{\langle t-1 \rangle}$ — shape $(n_h, 1)$ — previous hidden state
- $x^{\langle t \rangle}$ — shape $(n_x, 1)$ — current input
- $a^{\langle t \rangle}$ — shape $(n_h, 1)$ — new hidden state
- $y^{\langle t \rangle}$ and $\hat{y}^{\langle t \rangle}$ — shape $(n_y, 1)$ — target and prediction at the current time step

The cell has three weight matrices (plus bias vectors):

- $W_{aa}$ — shape $(n_h, n_h)$ — maps the previous hidden state to the new one
- $W_{ax}$ — shape $(n_h, n_x)$ — maps the current input to the hidden state
- $W_{ya}$ — shape $(n_y, n_h)$ — maps the hidden state to the output
- $b_a$ — shape $(n_h, 1)$ — hidden-state bias
- $b_y$ — shape $(n_y, 1)$ — output bias

**Stacking trick:** $W_{aa}$ and $W_{ax}$ are usually combined into a single matrix $W_a$ for efficiency:

$$
W_a = \big[\,W_{aa} \;\; W_{ax}\,\big] \quad \text{(horizontal stacking)}, \quad \text{shape } (n_h,\; n_h + n_x)
$$

This is paired with **vertical stacking** of the previous hidden state and current input into one column vector:

$$
\begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix}
$$

The stacked vector has shape $(n_h+n_x, 1)$, so the multiplication is valid:

$$
(n_h, n_h+n_x)(n_h+n_x, 1)=(n_h,1).
$$

So instead of computing two matrix-vector multiplications ($W_{aa}a^{\langle t-1\rangle}$ and $W_{ax}x^{\langle t\rangle}$) and adding them, you do **one** matrix-vector multiplication against the stacked vector — same result, tidier computation.

---

## 2. The RNN Cell

At every time step, the RNN cell does one simple thing: it takes the **hidden state carried over from the previous step** and the **input at the current step**, concatenates them, and pushes them through a fully-connected layer with a `tanh` activation to produce the new hidden state.

![RNN cell: the previous hidden state and current input are concatenated, transformed by a linear layer and tanh, and passed to both the next cell and a linear-plus-softmax prediction head.](assets/rnn_cell.png)

*The semicolon in $z = [a^{\langle t-1\rangle}; x^{\langle t\rangle}]$ denotes vertical stacking. The new hidden state $a^{\langle t\rangle}$ feeds both the next time step and the prediction head; softmax is shown for multi-class prediction.*

**In words:** the hidden state of the previous time step $a^{\langle t-1 \rangle}$ and the input of the current time step $x^{\langle t \rangle}$ are **concatenated** and fed into a fully-connected (FC) layer with a `tanh` activation function.

### Unrolled across time

Drawing the *same* cell repeated across time steps (it's the same weights reused, not different cells) shows how information flows forward through the sequence:

![RNN unrolled over three time steps: each input enters its cell, and each new hidden state feeds both the next cell and a linear-plus-softmax prediction head. All time steps share the same parameters.](assets/rnn_unrolled.png)

Each cell shares the **same** $W_{aa}$, $W_{ax}$, $W_{ya}$ (and biases) — the weights don't change across time steps, only the inputs and hidden state do.

---

## 3. The Forward Pass — Equations

**Step 1 — Compute the new hidden state:**

$$
a^{\langle t \rangle} = g_1\Big(W_{aa}\,a^{\langle t-1 \rangle} + W_{ax}\,x^{\langle t \rangle} + b_a\Big)
$$

Using the stacked-matrix trick from Section 1, this is equivalent to:

$$
a^{\langle t \rangle} = g_1\left(
W_a
\begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix}
+ b_a
\right)
$$

where the column vector is the vertical stacking described above.

**Step 2 — Compute the prediction:**

$$
\hat{y}^{\langle t \rangle} = g_2\Big(W_{ya}\,a^{\langle t \rangle} + b_y\Big)
$$

**Choice of activations:**
- $g_1$ = `tanh` (almost always, for the hidden state)
- $g_2$ = `sigmoid` (binary output) or `softmax` (multi-class output)

**Shape check:**

$$
\begin{aligned}
W_{aa}a^{\langle t-1 \rangle}&: (n_h,n_h)(n_h,1)=(n_h,1),\\
W_{ax}x^{\langle t \rangle}&: (n_h,n_x)(n_x,1)=(n_h,1),\\
W_{ya}a^{\langle t \rangle}&: (n_y,n_h)(n_h,1)=(n_y,1).
\end{aligned}
$$

Thus the two terms used to form the hidden state have matching shape $(n_h,1)$, and the prediction has shape $(n_y,1)$.

---

## 4. Loss Function

For a single binary output at one time step, use binary cross-entropy:

$$
\mathcal{L}^{\langle t \rangle}\big(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}\big) = -y^{\langle t \rangle}\log \hat{y}^{\langle t \rangle} - \big(1 - y^{\langle t \rangle}\big)\log\big(1 - \hat{y}^{\langle t \rangle}\big)
$$

For multi-class output, use categorical cross-entropy instead. The **total loss** for a sequence is the sum of the per-time-step losses:

$$
\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}^{\langle t \rangle}\big(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}\big)
$$

This total loss is what gets backpropagated through time (BPTT) to update $W_{aa}$, $W_{ax}$, $W_{ya}$, $b_a$, and $b_y$.

---

## 5. Quick Recap

1. Initialize $a_0$ as a vector of zeros.
2. At each time step $t$: concatenate $a^{\langle t-1\rangle}$ and $x^{\langle t\rangle}$, pass through a `tanh`-activated FC layer to get $a^{\langle t\rangle}$.
3. Pass $a^{\langle t\rangle}$ through a `softmax`/`sigmoid` layer to get the prediction $\hat{y}^{\langle t\rangle}$.
4. Repeat for every time step, reusing the **same** weights.
5. Sum the per-step losses to get the total sequence loss, and backpropagate through time.

**Key limitation to keep in mind:** because the same small set of weights is reused across many time steps, plain RNNs like this one struggle with long sequences — gradients tend to vanish or explode. This is the motivation for gated variants like GRUs and LSTMs, which add mechanisms to control what information is kept or forgotten over long sequences.
