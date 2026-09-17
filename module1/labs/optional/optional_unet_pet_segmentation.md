# Optional lab: Pet segmentation with U-Net

**Estimated time:** 2–3 hours  
**Prerequisites:** PyTorch, convolutional layers, encoder–decoder architectures, and basic model training.  
**Platform:** Kaggle Notebook with GPU enabled.

## Goal

Train a U-Net to segment the foreground pet in a photograph. For every input pixel, the model should predict whether it belongs to the pet or the background. This is a **semantic segmentation** task: unlike image classification, the model must make a prediction at every spatial location.

Use [The Oxford-IIIT Pet Dataset With Annotations](https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset). It contains pet photographs and pixel-level trimap annotations. The dataset is small enough for a Kaggle Notebook and has varied pet sizes, poses, lighting, and backgrounds.

## Learning objectives

By the end of the lab, you should be able to:

- prepare aligned image–mask pairs for a segmentation model;
- implement a U-Net encoder, decoder, and skip connections;
- explain how encoder downsampling increases receptive field;
- train a binary segmentation model and evaluate it with Dice and IoU;
- inspect qualitative predictions and identify failure cases.

## Part 1: Explore and prepare the data

1. Attach the dataset to a Kaggle Notebook and locate the pet images and their trimap masks.
2. Display at least five image–mask pairs. Verify that each mask has the same height and width as its image.
3. Create binary target masks: foreground pet = `1`, background = `0`.
   - The original trimaps also mark an uncertain boundary. For the core task, choose and document one rule: count the boundary as foreground, count it as background, or exclude it from the loss.
4. Make reproducible train, validation, and test splits, such as 80/10/10. Do not use the test split while selecting hyperparameters.
5. Resize images and masks to $128\times128$ or $160\times160$. Use bilinear interpolation for images and nearest-neighbour interpolation for masks.
6. Add modest augmentation to the training set, such as horizontal flips, small rotations, and brightness changes. Any geometric transformation must be applied identically to the image and its mask.

## Part 2: Build a small U-Net

Implement a U-Net with the following components:

- an encoder made of blocks containing two $3\times3$ convolutions followed by $2\times2$ max-pooling with stride 2;
- a bottleneck block at the lowest spatial resolution;
- a decoder that upsamples using transpose convolutions or interpolation followed by convolution;
- skip connections that concatenate encoder features with decoder features at the matching spatial resolution;
- a one-channel output layer that produces logits; apply sigmoid only when turning logits into mask probabilities for visualization or thresholding.

Start with channel widths such as 32, 64, 128, and 256. Confirm the tensor shape after each encoder and decoder block. If image dimensions cause a skip-connection mismatch, resize/crop consistently or choose dimensions divisible by $2^n$, where $n$ is the number of pooling levels.

### Receptive-field question

In your notebook, explain why the encoder gives the bottleneck features a larger receptive field. In particular, explain that $3\times3$ convolutions combine neighbouring features and that each stride-2 pooling operation doubles the spacing, measured in input pixels, between adjacent features at the next level. Explain why the skip connections are still needed: they restore high-resolution local details such as pet boundaries.

## Part 3: Train and evaluate

Train for 10–20 epochs. Begin with a short run of one epoch to validate the data pipeline and output shapes.

- Use `BCEWithLogitsLoss` (and omit the final sigmoid during training), or combine binary cross-entropy with Dice loss.
- Report validation Dice score and intersection over union (IoU). Do not rely only on pixel accuracy, because background pixels can dominate it.
- Save the model checkpoint with the best validation Dice score and evaluate it once on the held-out test set.

The metrics for binary masks are

$$
\operatorname{Dice}=\frac{2|P\cap G|}{|P|+|G|},
\qquad
\operatorname{IoU}=\frac{|P\cap G|}{|P\cup G|},
$$

where $P$ is the thresholded predicted mask and $G$ is the ground-truth mask. Add a small numerical constant to denominators in code to avoid division by zero.

## Deliverables

Submit one Kaggle Notebook containing:

1. Five inspected image–mask examples and a description of the trimap-to-binary conversion.
2. The U-Net implementation and shape checks.
3. A training/validation metric plot.
4. Dice and IoU on the held-out test set.
5. At least eight test visualizations, each showing the input image, ground-truth mask, and predicted mask.
6. A short discussion of two failure cases and one paragraph answering the receptive-field question.

## Optional extensions

Choose one extension after the core task works:

- Treat the trimap as a three-class segmentation problem: pet, background, and boundary.
- Compare a U-Net with and without skip connections. Which errors become more visible without them?
- Add dilated convolutions in the bottleneck and compare validation Dice/IoU with the baseline.
- Compare Dice loss, binary cross-entropy, and their combination.

## Practical Kaggle guidance

Use a GPU accelerator, a batch size of 16 or 32, and a fixed random seed. If training is slow, first develop with a reproducible subset of 2,000 training images, then run the final experiment on the full training split. Keep the test set fixed throughout the assignment.
