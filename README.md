# DCGAN for Fashion-MNIST Image Generation

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate synthetic 28x28 grayscale images of fashion items using the **Fashion-MNIST** dataset. The model is built with **TensorFlow** and **Keras**, utilizing a custom training loop with `tf.GradientTape`.

## 🚀 Overview
Generative Adversarial Networks (GANs) consist of two models—a **Generator** and a **Discriminator**—that compete against each other.
- **The Generator** learns to create realistic images from random noise.
- **The Discriminator** learns to distinguish between real images from the dataset and fake images created by the generator.

## 📊 Dataset
The project uses the **Fashion-MNIST** dataset, which consists of 70,000 grayscale images in 10 categories. The images are 28x28 pixels.
- Data is normalized to the range `[-1, 1]` to match the `tanh` activation function of the generator's output layer.

## 🏗️ Architecture
### Generator
- Starts with a Dense layer that takes a 100-dimensional noise vector.
- Uses **Conv2DTranspose** (fractionally-strided convolutions) to upsample to 28x28.
- Employs **BatchNormalization** and **LeakyReLU** activation.
- Final layer uses `tanh` activation.

### Discriminator
- A standard CNN-based classifier.
- Uses **Conv2D** layers with strided convolutions to downsample images.
- Employs **LeakyReLU** and **Dropout** (recommended for stability).
- Final layer uses `sigmoid` activation to output a probability.

## 🛠️ Installation
To run this project, you need Python 3.x and the following libraries:

```bash
pip install tensorflow numpy matplotlib imageio
