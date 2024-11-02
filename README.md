# Face Generation, using DCGAN (Deep Convolutional Generative Adversarial Networks)

# Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) are a class of machine learning systems introduced by Ian Goodfellow and his
colleagues in 2014. They consist of two neural networks that work against each other in an adversarial process:

1. Generator: Creates synthetic data (like images) by trying to fool the discriminator
2. Discriminator: Attempts to distinguish between real and generated data

The process works like this:

- The generator creates fake samples
- The discriminator evaluates both real and fake samples
- Both networks improve through this competition - the generator gets better at creating realistic data, while the
  discriminator gets better at detecting fakes

GANs have found numerous applications:

- Creating realistic images, videos, and audio
- Data augmentation for training other AI models
- Converting sketches to photorealistic images
- Style transfer between images
- Creating synthetic data for training when real data is limited

Some challenges with GANs include:

- Training instability
- Mode collapse (generator produces limited varieties)
- Difficulty measuring performance
- Computational intensity

Despite these challenges, GANs have become fundamental to many AI applications, particularly in computer vision and
content generation. They've led to technologies like deepfakes and AI art generators.

In this course, we will consider a specific type of deep learning model**:**

Generative Adversarial Networks, or GANs.
Applications of GANs include:

Image generation
Image to image translation
"Deep fakes"

In this course, we will cover the following topics:

Generative Adversarial Networks

1. Build a generator network
2. Build a discriminator network
3. Build GAN losses
4. Train on the MNIST dataset

Training a Deep Convolutional GANs

1. Build a DCGAN generator and discriminator
2. Train a DCGAN model on the CIFAR10 dataset
3. Implement GAN evaluation metrics

Image to Image Translation

1. Implement CycleGAN dataloaders
2. Implement CycleGAN generator and loss functions
3. Train a CycleGAN

Modern GANs: WGAN-GP, ProGAN and StyleGAN

1. Implement the Wasserstein Loss with gradient penalties
2. Implement a ProGAN model
3. Implement the components of a StyleGAN model

Image Generation
Fully Visible Belief Networks – where the model generates an image one pixel at a time. This is also called an
Autoregressive Model.

Generative Adversarial Networks (GANs) – where the model generates an entire image in parallel using a differentiable
function

How to Get Realistic Images
GANs used a combination of neural networks to accomplish the task of image generation:

Generator Network – takes random input through a differentiable function to transform and reshape it to have a
recognizable structure. The output is a realistic image.

Unlike training a supervised learning model, when training a generator model, there is no classification/label to
associate with each image. It creates additional images based on a probability distribution.

Discriminator Network – is a regular neural net classifier that learns to guide the generator network by outputting the
probability that the input is real. Fake images are 0 and real images are 1.
The generator network is forced to produce more realistic images to "fool" the discriminator network.

### Quiz Question

| Network       | Description                                    | Explanation                                                                                                                                                 |
|---------------|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Generator     | Takes random noise as input.                   | The Generator starts with random noise (latent space) as its input. This is the first step in the GAN process.                                              |
| Generator     | Transforms noise into a realistic image.       | The Generator's job is to convert the random noise into synthetic data (like images) that looks real. It learns to create increasingly convincing fakes.    |
| Discriminator | Learns to classify real and fake images.       | The Discriminator is trained to distinguish between real images from the training set and fake images created by the Generator.                             |
| Discriminator | Outputs the probability that an input is real. | The Discriminator produces a probability score (typically between 0 and 1) indicating how likely it thinks the input is a real rather than generated image. |

The pairing works because:

- The Generator has two connected functions: taking noise input and transforming it into realistic output
- The Discriminator has two connected functions: learning to classify images and outputting probability scores
- Together they form an adversarial relationship where:
    - The Generator tries to create better fakes
    - The Discriminator tries to get better at detecting fakes
    - This competition drives both networks to improve

These matches highlight the core mechanics of how GANs work as a two-player adversarial game.

### Games and Equilibria: Part 1

Adversarial
In GANs, adversarial means that two networks, the generator and the discriminator, compete with each other for improved
image generation.

This "competition" between the networks is based on Game Theory.

Game Theory – a form of applied mathematics used to model cooperation and conflict between rational agents in any
situation

### Games and Equilibria: Part 2

Equilibria and GANs
Most ML models are based on optimization and follow the general pattern of

Determine model parameters
Have a cost function of these parameters
Minimize the cost
GANs are different because there are two players, the generator and the discriminator, and each player has its own cost.
The "game" is therefore defined by a value function.

The generator wants to minimize the value function.
The discriminator wants to maximize the value function.
The saddle point is when equilibrium is reached, a point in the parameters of both players that is simultaneously a
local minimum for each player's costs with respect to that player's parameters.
A key learning problem for GANs is finding the equilibrium of a game involving cost functions that are:

1. High dimensional
2. Continuous
3. Non-convex

Let me break down each statement and explain whether it's true or false:

1. Statement: "Training a GAN is equivalent to playing a game where both networks are competing with each other."

- Answer: TRUE
- Explanation: This is an accurate description of GANs. The generator and discriminator are indeed in a minimax
  game/competition where the generator tries to create convincing fakes while the discriminator tries to detect them.
  This adversarial relationship is fundamental to how GANs work. Training GAN is very challenging because of their
  unstable nature! Thankfully, researchers came up with a lot of different techniques to make GAN training easier!

2. Statement: "When training a GAN, after some time, the equilibrium is always reached."

- Answer: FALSE
- Explanation: GANs don't always reach equilibrium. They can suffer from various training issues like mode collapse,
  vanishing gradients, or failure to converge. Reaching equilibrium is not guaranteed and is actually one of the
  challenging aspects of training GANs.

3. Statement: "Game Theory is a subfield of deep learning invented to describe GANs training."

- Answer: FALSE
- Explanation: This is incorrect. Game theory is a much older field of mathematics and economics that existed long
  before deep learning or GANs. It wasn't invented for GANs - rather, GAN training was described using existing game
  theory concepts.

4. Statement: "The GAN equilibrium is reached when the discriminator loss reaches 0"

- Answer: FALSE
- Explanation: This is incorrect. The ideal GAN equilibrium is reached when the discriminator's output is 0.5 (
  indicating 50% uncertainty between real and fake), not when the loss is 0. A discriminator loss of 0 would actually
  indicate a failing GAN where the discriminator has become too powerful.

Here's an explanation of GANs through game theory:

# GANs as a Two-Player Game

## The Players and Their Objectives

1. **Generator (G)**

- Role: Creates fake data from random noise
- Goal: Minimize the value function
- Strategy: Tries to fool the discriminator by producing increasingly realistic data

2. **Discriminator (D)**

- Role: Classifies data as real or fake
- Goal: Maximize the value function
- Strategy: Tries to correctly distinguish between real and generated data

## The Game Dynamics

### Value Function Mechanics

- G and D are playing a minimax game
- Think of it like a counterfeiter (G) vs detective (D) game:
    - G tries to create better counterfeits
    - D tries to get better at detecting counterfeits

### Equilibrium Concept

- Saddle point: The theoretical optimal point where:
    - G creates such realistic data that
    - D can only achieve 50% accuracy (like random guessing)

## Training Challenges

The game is complex because the value function is:

1. **High dimensional**: Many parameters to optimize
2. **Continuous**: Not discrete choices but continuous adjustments
3. **Non-convex**: Multiple local optima exist

Unlike traditional ML models that just minimize one cost function, GANs must balance two competing objectives, making
equilibrium hard to achieve in practice.

### Key Difference from Traditional ML

```textmate
Traditional ML:
- Single cost function
- Simple minimization
- Clear optimization path

GANs:
- Two competing costs
- Minimax optimization
- Complex equilibrium search
```

This game theoretic framework helps explain both the power and the training difficulties of GANs.

### Tips for Training GANs: Part 1

Good Architecture
Fully Connected Architecture can be used for simple tasks that meet the following criteria:

No convolution
No recurrence
The generator and discriminator have a least one hidden layer
Leaky ReLU helps to make sure that the gradient can flow through the entire architecture and is a popular choice for
hidden layer activation functions.

The Hyperbolic Tangent activation function is a popular output choice for the generator and means data should be scaled
to the interval from -1 to +1.

A Sigmoid Unit is used to enforce the constraint that the output of the discriminator is a probability.

Design Choice
One of the design choices from the DCGAN architecture is Adam, an optimization algorithm.

A common error is that people forget to use a numerically stable version of cross-entropy, where the loss is computed
using the logits.

Logits – the values produced by the discriminator right before the sigmoid.
Tips for Training
A simple trick is to multiply the 0 or 1 labels by a number a bit less than 1. This is a GANs-specific label smoothing
strategy similar to that used to regularize normal classifiers.
For the generator loss, minimize cross-entropy with the labels flipped.

### Tips for Training GANs: Part 2

Scaling GANs
Convolutional Neural Networks (CNN) are needed to scale GANs to work on larger images. Scaling GANs relies on an
understanding of:

Classifier Convolutional Net – starting with a tall and wide feature map and moving to very short and narrow feature
maps
Generator Net – starting with short and narrow feature maps and moving to a wide and tall image
Batch Normalization – on potentially every layer except the output layer of the generator and the input layer of the
discriminator

**Question 1**: Match each activation with its corresponding property

Q1 Matches:

- Hyperbolic tangent → output bounded between -1 and 1
- Leaky ReLU → helps the generator learn
- Sigmoid → output bounded between 0 and 1

Explanation:

- Hyperbolic tangent (tanh) squashes inputs to range [-1,1], making it useful for outputs that need both positive and
  negative values
- Leaky ReLU prevents "dying ReLU" problem common in generators by allowing small negative values to pass through
- Sigmoid squashes inputs to range [0,1], making it useful for probability outputs

Let me explain these three activation functions in detail:

1. **Hyperbolic Tangent (tanh)**

- Mathematical form: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- Properties:
    - Output range: [-1, 1]
    - Zero-centered: Outputs are symmetric around 0
    - S-shaped curve (similar to sigmoid but centered at 0)
- Use cases:
    - Hidden layers in neural networks
    - When outputs need to be normalized between -1 and 1
    - Common in GANs' generator output layer

2. **Leaky ReLU**

- Mathematical form: f(x) = x if x > 0; αx if x ≤ 0 (where α is a small constant, typically 0.01)
- Properties:
    - Allows small negative values (doesn't completely zero them out)
    - Prevents "dying ReLU" problem where neurons can get stuck
    - Has a non-zero gradient for negative inputs
- Benefits for Generator:
    - Helps maintain gradient flow
    - Prevents dead neurons
    - Allows better learning of features in negative space

3. **Sigmoid**

- Mathematical form: σ(x) = 1/(1 + e^(-x))
- Properties:
    - Output range: [0, 1]
    - S-shaped curve
    - Often used for binary classification
    - Outputs can be interpreted as probabilities
- Use cases:
    - Final layer in binary classification
    - Discriminator output in GANs
    - When output needs to be interpreted as probability

Comparison:

```textmate
Function        | Output Range | Gradient Properties
----------------|--------------|--------------------
tanh            | [-1, 1]     | Can saturate at extremes
Leaky ReLU      | (-∞, ∞)     | Small gradient for negative values
Sigmoid         | [0, 1]      | Can saturate at extremes
```

**Question 2**: Which of the following statements are true? (Multiple correct choices)

Answers (All true):

1. "The binary cross entropy (BCE) loss can be used to train both the generator and the discriminator."
2. "Label smoothing consists in multiplying the labels by a float smaller than 1."
3. "When training the generator, the BCE loss is used with flipped labels (fake = 1, real = 0)"
4. "The BCE loss is only one of the possible loss functions to train a GAN. Many other options exist."

Explanation:

- BCE loss is versatile and can be used for both networks in a GAN
- Label smoothing helps prevent overconfident predictions by making target values less extreme (e.g., 0.9 instead of
  1.0)
- Generator training uses flipped labels to maximize the probability of discriminator being wrong
- While BCE is common, other loss functions like Wasserstein loss or least squares loss can also be used for GANs

### MNIST GAN

The steps for building a GAN to generate new images can be summarized as follows:

1. Create a classifier by training on dataset images
2. Create an adversarial training using a discriminator and generator
    - The discriminator acts as a simple classifier distinguishing between real and fake images
    - The generator acts as an adversary with the goal of tricking the discriminator into tagging generated images as "
      real"

3. Define generator and discriminator networks with opposing goals and loss functions


1. **Data Sources**:

- Top Path: MNIST training data (real handwritten digits)
    - Shows a grid of handwritten numbers
    - A sample '8' is extracted as real data
- Bottom Path: Latent sample (z)
    - Random noise input
    - Shown as a pixelated grid

2. **Generator Network**:

- Takes the latent sample (random noise) as input
- Transforms noise into a fake digit image
- In the example, tries to generate an '8'
- Output looks similar to but not exactly like real MNIST digits

3. **Discriminator Network**:

- Receives two types of inputs:
    - Real samples from MNIST dataset
    - Fake samples from Generator
- Outputs a value around 0.5 when uncertain
- Goal: Distinguish real from fake images

4. **Opposing Loss Functions**:

- Generator's goal:
    - Make fake images that fool the discriminator
    - Wants discriminator to output 0.5 (uncertain)
- Discriminator's goal:
    - Correctly classify real vs fake
    - Output 1 for real images
    - Output 0 for fake images

5. **Training Process**:

- Generator improves at creating realistic digits
- Discriminator improves at detection
- Success is reached when discriminator outputs 0.5 (can't tell difference)
- Both networks have opposing objectives, creating an adversarial game

This creates a feedback loop where both networks continually improve until the generated images become indistinguishable
from real MNIST digits.

<br>

![localImage](/images/workflow.png)

<br>

**Game Setup: Generator vs Discriminator**

1. **Training Data (MNIST)**:

- Discriminator learns from real handwritten digits (ground truth)
- Used as reference point for what "real" data looks like
- In image: Grid of actual MNIST digits with sample "8"

2. **The Game Dynamic**:

**Player 1: Generator (G)**

- Strategy: Create fake digits from random noise (latent space z)
- Objective: Fool D by generating images that look like real MNIST digits
- Learning Process:
    - Gets better at mimicking real digit characteristics
    - Adapts based on how well it fools D
    - Success = D cannot distinguish its output from real digits

**Player 2: Discriminator (D)**

- Strategy: Classify images as real or fake
- Objective: Correctly identify both:
    - Real digits from MNIST dataset
    - Fake digits from Generator
- Learning Process:
    - Gets better at spotting subtle differences
    - Updates its detection based on G's improvements
    - Success = Accurate classification of both real and fake

**The Adversarial Game**:

1. G generates fake digit (from noise)
2. D receives both real MNIST digits and G's fakes
3. D tries to distinguish them
4. G improves based on D's success
5. D improves based on G's deception
6. Equilibrium reached when D outputs 0.5 (complete uncertainty)

This creates an arms race where:

- As G gets better at faking, D must get better at detecting
- As D gets better at detecting, G must get better at faking
- Both continuously improve through competition

Let me clarify how each network uses training data:

**Discriminator (D)**:

- ✅ Uses training data directly
- Learns from two sources:
    1. Real MNIST digits (training data)
    2. Fake digits from generator
- Needs training data to learn what "real" looks like
- Uses this knowledge to distinguish fake from real

**Generator (G)**:

- ❌ Does NOT use training data directly
- Only receives:
    1. Random noise (latent space z) as input
    2. Feedback from discriminator's decisions
- Never sees actual MNIST digits
- Learns indirectly through D's feedback
- Improves based on how well it fools D

This is a key aspect of GANs:

- G learns to generate realistic data without ever seeing real data
- G only knows if its outputs are "good" or "bad" based on D's feedback
- D acts as a teacher for G through the adversarial process
- Training data is only used by D to learn what "real" looks like

The training data informs D's standards, which then indirectly guides G's learning through their adversarial
relationship.

**Generator (G)**:

- Does use training data INDIRECTLY through the learning process
- Not by seeing it directly, but through:
    1. Network architecture designed for the specific type of data (e.g., images of cats)
    2. Loss function based on discriminator's feedback
    3. Backpropagation that updates G's parameters

**Example with Cat GAN**:

1. Training Process:
    - D learns from real cat images what cats look like
    - D provides feedback to G about how "cat-like" its outputs are
    - G's parameters get updated to better match the distribution of real cat data
    - The whole network is structured to generate cat-like images

2. Network Design:
    - G's architecture is specifically designed for the target domain (cats)
    - Layer structure matches the complexity of cat images
    - Output dimensions match cat image dimensions

So you're correct:

- G does need domain-specific knowledge
- This comes from:
    - Network architecture
    - Training process
    - Indirect learning through D's feedback
- While G doesn't directly see training data, the entire system is trained on it

It's more accurate to say G learns the distribution of the training data rather than directly copying it.

Let me break this down into a question and answer format:

**Question**: Match each component of a GAN with its function.

**Answer**:

1. Latent vector → Random noise vector fed through the generator
2. Generator → Turns noise into realistic images
3. Discriminator → Classifies real from fake images
4. Real images → Distribution the generator is trying to replicate
5. Binary Cross Entropy → Loss function used to evaluate the discriminator's performance

GANs are made of two networks, the generator and the discriminator. The generator generates realistic images from random
noise and tries to fool the discriminator, whose job is to separate real from fake images.

**Explanation**:

Each component serves a specific purpose in the GAN architecture:

1. **Latent vector**

- Input noise that seeds the generation process
- Random values that get transformed into meaningful data

2. **Generator**

- Neural network that transforms random noise
- Learns to create realistic images from random input

3. **Discriminator**

- Acts as a binary classifier
- Learns to distinguish between real and generated images

4. **Real images**

- Training data that represents the target distribution
- What the generator is trying to learn to replicate

5. **Binary Cross Entropy**

- Loss function that guides training
- Measures how well discriminator distinguishes real from fake
- Helps both networks improve through training

These components work together in an adversarial process to generate realistic data.

# Training a Deep Convolutional GANs

## Introduction to DCGANs

Welcome to this lesson on Deep Convolutional Generative Adversarial Networks, also known as, DCGANs.

By the end of this lesson, you will be able to:

1. Build a convolutional generator based on the DCGAN model
2. Build a convolutional discriminator also based on the DCGAN model
3. Train these networks on a dataset of RGB images
4. Implement two metrics for GANs performance evaluation

<br>

![localImage](/images/DCGAN.png)

<br>


In this lesson on Deep Convolutional GANs (DCGANs) we will cover the following topics:

Build DCGAN Discriminator and Generator
Train a DCGAN Model
Evaluate GANs using New Metrics

Understanding a DCGAN
In this lesson, you will be training a GAN on the CIFAR10 dataset,(opens in a new tab) a labeled subset of the 80
million tiny images dataset(opens in a new tab).

To improve model performance, convolutional layers will be used to make a DCGAN.

1. DCGANs have generator and discriminator networks, but the networks are made of convolutional layers that are designed
   to
   work with spatial data
2. The discriminator will be a convolutional neural network (CNN) that classifies images are real or fake
3. The generator will be a transpose CNN that upsamples a latent vector z and generates realistic images that can fool
   the
   discriminator.

### DCGAN Discriminator

The DCGAN Discriminator is:

1. A convolutional neural network (CNN) with one fully connected layer at the end
2. There are no max-pooling layers in the network
3. Down-sampling is accomplished using convolutional layers that have a stride equal to 2
4. Batch normalization and Leaky ReLU activations are applied to the outputs of all hidden layers
5. After a series of downsampling convolutional layers, the final layer is flattened and connected to a single sigmoid
   unit
6. The sigmoid unit output has a range from 0 to 1, indicating if an image is "real" or "fake"

<br>

![localImage](/images/DCGAN_Discriminator.png)

<br>


Leaky ReLu – a function that will reduce any negative values by multiplying those values by some small coefficient,
known as the negative slope.

Batch Normalization – scales the layer outputs to have a mean of 0 and variance of 1, to help the network train faster
and reduce problems due to poor parameter initialization.

**Quiz Question**: Which statements are true about the DCGAN discriminator?

**Answers**:

1. "It uses Leaky ReLU activation instead of ReLU"

- Answer: True
- Explanation: DCGAN discriminator uses Leaky ReLU to prevent dying ReLU problem and allow better gradient flow for
  negative values, which is crucial for the discriminator's learning process.

2. "It uses convolution layer to decrease the input spatial resolution"

- Answer: True
- Explanation: DCGAN discriminator uses strided convolutions to progressively decrease spatial dimensions of the input,
  allowing it to process and analyze images at different scales.

3. "It also uses pooling layers to decrease the input spatial resolution"

- Answer: False
- Explanation: DCGAN specifically avoids using pooling layers, instead relying on strided convolutions for downsampling.
  This architectural choice helps maintain spatial information and improve training stability.

**Key Points**:

- DCGAN discriminator uses an architecture optimized for image processing
- It relies on strided convolutions rather than pooling
- Leaky ReLU helps prevent the vanishing gradient problem
- The design choices aim to create a stable and effective discriminator for image generation tasks

### DCGAN Generator

The task of a DCGAN Generator is to understand patterns in the underlying structure and features of the training data,
in ways that allow it to create realistic generated images.

The DCGAN Generator:

1. Has an input, random vector z
2. Has an image output that can be sent to the discriminator
3. Up-samples the vector z until it is the same shape as the training images
4. Uses transposed convolutions
5. ReLU activations and batch normalization is used on all hidden layers
6. A tanh activation function is applied the outputs of the final layer

<br>

![localImage](/images/DCGAN_Generator.png)

DCGAN Generator does NOT have direct access to the training data. Here's how it actually works:

**Generator's Learning Process**:

1. Never sees real training data directly
2. Learns indirectly through:
    - Discriminator's feedback
    - Backpropagation of gradients
    - Loss function signals

**How Generator Learns Patterns**:

1. **Input**: Takes random noise (latent vector)
2. **Process**:
    - Generates fake images
    - Gets feedback from discriminator about how "real" they look
    - Updates its parameters based on this feedback

3. **Learning Chain**:
    - Discriminator sees real training data
    - Discriminator learns what "real" looks like
    - Generator gets feedback from discriminator
    - Generator improves based on this indirect feedback

**Key Point**:

- The generator learns to understand patterns and features indirectly
- It's like an artist learning to paint without seeing real paintings, but getting feedback from an art critic who has
  seen them
- The discriminator acts as a "teacher" that has studied the real data and guides the generator's learning

This indirect learning process is a key characteristic of GANs, where the generator improves through adversarial
feedback rather than direct access to training data.

Generating Images
To generate an image, the generator:

1. Connects the input vector z to a fully connected layer
2. The fully connected layer is reshaped to a 4x4 XY shape of a given depth
3. A stack of larger layers is built by upsampling with transpose convolution
4. Each layer is doubled in XY size using strides of 2, and depth is reduced
5. The final output is a generated image the same size as the training images

**Quiz Question**: Which statements are true about the DCGAN generator?

**Answers**:

1. "It only uses fully connected layers"

- Answer: False
- Explanation: DCGAN generator uses both convolutional layers and transpose convolutions (deconvolutions) to generate
  images, not just fully connected layers. This architecture is specifically designed for image generation.

2. "It outputs an RGB image in the -1/1 range"

- Answer: True
- Explanation: The DCGAN generator's final layer typically uses tanh activation, which outputs values in the
  range [-1, 1], making it suitable for normalized RGB image generation.

3. "It uses transpose convolution layer to progressively increase the resolution"

- Answer: True
- Explanation: DCGAN generator uses transpose convolutions (sometimes called deconvolutions) to progressively upscale
  the spatial dimensions from the initial latent vector to the final image size.

4. "It uses Leaky ReLU activation instead of ReLU"

- Answer: False
- Explanation: DCGAN generator typically uses regular ReLU activations, not Leaky ReLU. Leaky ReLU is more commonly used
  in the discriminator. The generator benefits from the standard ReLU's properties.

**Key Points**:

- Uses a combination of layers, not just fully connected
- Outputs normalized images using tanh
- Uses transpose convolutions for upscaling
- Uses regular ReLU activations

### Batch Normalization

Batch normalization was introduced in Sergey Ioffe's and Christian Szegedy's 2015 paper Batch Normalization:
Accelerating Deep Network Training by Reducing Internal Covariate Shift(opens in a new tab). The idea is that, instead
of just normalizing the inputs to the network, we normalize the inputs to every layer within the network.

Batch Normalization
It's called "batch" normalization because, during training, we normalize each layer's inputs by using the mean and
standard deviation (or variance) of the values in the current batch. These are sometimes called the batch statistics.

Specifically, batch normalization normalizes the output of a previous layer by subtracting the batch mean and dividing
by the batch standard deviation.

Why might this help? Well, we know that normalizing the inputs to a network helps the network learn and converge to a
solution. However, a network is a series of layers, where the output of one layer becomes the input to another. That
means we can think of any layer in a neural network as the first layer of a smaller network.

Normalization at Every Layer
For example, imagine a 3 layer network.

<br>

![localImage](/images/bn_1.png)

<br>


Instead of just thinking of it as a single network with inputs, layers, and outputs, think of the output of layer 1 as
the input to a two layer network. This two layer network would consist of layers 2 and 3 in our original network.


<br>

![localImage](/images/bn_2.png)

<br>

Likewise, the output of layer 2 can be thought of as the input to a single layer network, consisting only of layer 3.


<br>

![localImage](/images/bn_3.png)

<br>


When you think of it like this - as a series of neural networks feeding into each other - then it's easy to imagine how
normalizing the inputs to each layer would help. It's just like normalizing the inputs to any other neural network, but
you're doing it at every layer (sub-network).

Internal Covariate Shift
Beyond the intuitive reasons, there are good mathematical reasons to motivate batch normalization. It helps combat what
the authors call internal covariate shift.

In this case, internal covariate shift refers to the change in the distribution of the inputs to different layers. It
turns out that training a network is most efficient when the distribution of inputs to each layer is similar!

And batch normalization is one method of standardizing the distribution of layer inputs. This discussion is best handled
the paper on Batch Normalization (ArXiv PDF)(opens in a new tab) and in Deep Learning(opens in a new tab), a book you
can read online written by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Specifically, check out the batch
normalization section of Chapter 8: Optimization for Training Deep Models(o

The Math
Next, let's do a deep dive into the math behind batch normalization. This is not critical for you to know, but it may
help your understanding of this whole process!

Getting the mean and variance
In order to normalize the values, we first need to find the average value for the batch. If you look at the code, you
can see that this is not the average value of the batch inputs, but the average value coming out of any particular layer
before we pass it through its non-linear activation function and then feed it as an input to the next layer.

We represent the average as $\mu_B$ which is simply the sum of all of the values, $x_i$ divided by the number of values,
$m$:

$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$

We then need to calculate the variance, or mean squared deviation, represented as $\sigma_B^2$.
If you aren't familiar with statistics, that simply means for each value $x_i$, we subtract the average value (
calculated earlier as $mu_B$), which gives us what's called the "deviation" for that value. We square the result to get
the squared deviation. Sum up the results of doing that for each of the values, then divide by the number of values,
again $m$, to get the average, or mean, squared deviation.

$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m(x_i - \mu_B)^2$

Normalizing output values

Once we have the mean and variance, we can use them to normalize the values with the following equation. For each value,
it subtracts the mean and divides by the (almost) standard deviation. (You've probably heard of standard deviation many
times, but if you have not studied statistics you might not know that the standard deviation is actually the square root
of the mean squared deviation.)

$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

Above, we said "(almost) standard deviation". That's because the real standard deviation for the batch is calculated by
$\sqrt{\sigma_B^2}$, but the above formula adds the term epsilon before taking the square root. The epsilon can be any
small, positive constant, ex. the value 0.001. It is there partially to make sure we don't try to divide by zero, but it
also acts to increase the variance slightly for each batch.

Why add this extra value and mimic an increase in variance? Statistically, this makes sense because even though we are
normalizing one batch at a time, we are also trying to estimate the population distribution – the total training set,
which itself an estimate of the larger population of inputs your network wants to handle. The variance of a population
is typically higher than the variance for any sample taken from that population, especially when you use a small sample
size (a small sample is more likely to include values near the peak of a population distribution), so increasing the
variance a little bit for each batch helps take that into account.

At this point, we have a normalized value, represented as $\hat{x}_i$. But rather than use it directly, we multiply it
by a gamma value, and then add a beta value. Both gamma and beta are learnable parameters of the network and serve to
scale and shift the normalized value, respectively. Because they are learnable just like weights, they give your network
some extra knobs to tweak during training to help it learn the function it is trying to approximate.

$y_i = \gamma\hat{x}_i + \beta$

We now have the final batch-normalized output of our layer, which we would then pass to a non-linear activation function
like sigmoid, tanh, ReLU, Leaky ReLU, etc. In the original batch normalization paper, they mention that there might be
cases when you'd want to perform the batch normalization after the non-linearity instead of before, but it is difficult
to find any uses like that in practice.

### Benefits of Batch Normalization

Adding Batch Normalization Layers to a PyTorch Model
In the last notebook, you saw how a model with batch normalization applied reached a lower training loss and higher test
accuracy! There are quite a few comments in that code, and I just want to recap a few of the most important lines.

To add batch normalization layers to a PyTorch model:

1. You add batch normalization to layers inside the__init__ function.
2. Layers with batch normalization do not include a bias term. So, for linear or convolutional layers, you'll need to
   set
   bias=False if you plan to add batch normalization on the outputs.
3. You can use PyTorch's BatchNorm1d(opens in a new tab) function to handle the math on linear outputs or BatchNorm2d(
   opens
   in a new tab) for 2D outputs, like filtered images from convolutional layers.
4. You add the batch normalization layer before calling the activation function, so it always goes layer > batch norm >
   activation.

Finally, when you tested your model, you set it to .eval() mode, which ensures that the batch normalization layers use
the populationrather than the batch mean and variance (as they do during training).


<br>

![localImage](/images/batch_normalization.png)

<br>

The takeaway
By using batch normalization to normalize the inputs at each layer of a network, we can make these inputs more
consistent and thus reduce oscillations that may happen in gradient descent calculations. This helps us build deeper
models that also converge faster!

Take a look at the PyTorch BatchNorm2d documentation(opens in a new tab) to learn more about how to add batch
normalization to a model, and how data is transformed during training (and evaluation).

### Benefits of Batch Normalization

Batch normalization optimizes network training. It has been shown to have several benefits:

1. Networks train faster – Each training iteration will actually be slower because of the extra calculations during the
   forward pass and the additional hyperparameters to train during back propagation. However, it should converge much
   more quickly, so training should be faster overall.
2. Allows higher learning rates – Gradient descent usually requires small learning rates for the network to converge.
   And as networks get deeper, their gradients get smaller during back propagation so they require even more iterations.
   Using batch normalization allows us to use much higher learning rates, which further increases the speed at which
   networks train.
3. Makes weights easier to initialize – Weight initialization can be difficult, and it's even more difficult when
   creating deeper networks. Batch normalization seems to allow us to be much less careful about choosing our initial
   starting weights.
4. Makes more activation functions viable – Some activation functions do not work well in some situations. Sigmoids lose
   their gradient pretty quickly, which means they can't be used in deep networks. And ReLUs often die out during
   training, where they stop learning completely, so we need to be careful about the range of values fed into them.
   Because batch normalization regulates the values going into each activation function, non-linearlities that don't
   seem to work well in deep networks actually become viable again.
5. Simplifies the creation of deeper networks – Because of the first 4 items listed above, it is easier to build and
   faster to train deeper neural networks when using batch normalization. And it's been shown that deeper networks
   generally produce better results, so that's great.
6. Provides a bit of regularization – Batch normalization adds a little noise to your network. In some cases, such as in
   Inception modules, batch normalization has been shown to work as well as dropout. But in general, consider batch
   normalization as a bit of extra regularization, possibly allowing you to reduce some of the dropout you might add to
   a network.
7. May give better results overall – Some tests seem to show batch normalization actually improves the training results.
   However, it's really an optimization to help train faster, so you shouldn't think of it as a way to make your network
   better. But since it lets you train networks faster, that means you can iterate over more designs more quickly. It
   also lets you build deeper networks, which are usually better. So when you factor in everything, you're probably
   going to end up with better results if you build your networks with batch normalization.

### Optimization Strategy / Hyperparameters

Two Times Update Rule [TTUR]
Another approach for better GAN convergence consists in using the Two Times Update Rule (TTUR)(opens in a new tab). This
approach consists in running more update steps for the discriminator than for the generator. For example, for each
update of the generator, we run 3 updates of the discriminator.

Another way is to have a slightly higher learning rate for the discriminator. An example of TTUR can be found in the
official implementation by the Institute of Bioinformatics, Johannes Kepler University Linz(opens in a new tab).

A recipe for training Neural Networks(opens in a new tab)
The above link is a great resource to use when debugging neural networks. It applies to any type of deep learning model,
including GAN and was written by Andrej Karpathy, the head of AI at Tesla. Definitely a recommended read!

**Question 1**: Why is training GAN hard? (Check all correct choices)

**Answers**:

1. "The discriminator's job is much easier and it can easily overcome the generator"

- Answer: True
- Explanation: Discriminator can often learn too quickly and provide insufficient feedback to the generator, leading to
  training imbalance.

2. "GANs are harder to monitor because fluctuating losses are not a sign that the sign is going poorly"

- Answer: True
- Explanation: Unlike traditional neural networks, loss values in GANs don't reliably indicate training progress since
  both networks are competing.

3. "The minimax game is inherently hard because the equilibrium between generator and discriminator requires solving a
   hard optimization problem"

- Answer: True
- Explanation: Finding the Nash equilibrium between two competing networks is a complex optimization problem that's
  mathematically challenging.

**Question 2**: Finish the sentence for each concept

**Matches**:

1. "Dropout" → Adds regularization to the discriminator
2. "Label smoothing" → Consists in multiplying the target label by factor < 1
3. "Increasing the generator complexity" → Is helpful if the generated samples are not realistic enough
4. "Starting with a low learning rate and the Adam optimizer" → Is a safe way to get started when training a new
   architecture

Training GAN is challenging, but regularization techniques and default optimizer values can go a long way.

**Explanation**:

- Dropout prevents discriminator from becoming too strong
- Label smoothing helps prevent overconfident predictions
- Complex generator architecture helps capture detailed patterns
- Conservative optimization approach helps stabilize initial training

### GAN Evaluation

The Inception Model
The Inception Model is a concatenation of the outputs of layers of different filter sizes that allows deeper networks.
The Inception Score and the Frechet Inception use the Inception Model for their calculations.


<br>

![localImage](/images/inception_model.png)

<br>

Kullback Leibler (KL) Divergence
The KL divergence is a measure of distance between two probability distributions.

Low KL divergence means that the distributions are similar
High KL divergence means that they are different


<br>

![localImage](/images/kl.png)

<br>

### The Inception Score

The Inception Score leverages the KL divergence and the inception model to evaluate generated samples. To calculate the
inception score, build two probability distributions.

1. Conditional Distribution – feed a generated sample through the inception model pre-trained on the ImageNet dataset(
   opens in a new tab). The inception model will output a probability distribution with the last soft-max layer.
2. Marginal Distribution – the mean of all the p(y∣x) over all of the x values.
3. Use KL Divergence to measure the distance between the two distributions.

<br>

![localImage](/images/Inception.png)

<br>


The Inception Score is:

$e^{E[KL(p(y|x),p(y))]}$

where the exponent is the expected value of KL divergence between the marginal and the conditional label distribution
over all the generated samples.

Inception Score Limitations
The inception score is a great tool to calculate a GAN performance but has some limitations:

1. It relies on a model pre-trained on the ImageNet dataset.
2. It does not take into account the real dataset, which can be limiting in some situations.

### Frechet Inception Distance

Task 1:
**Quiz Question**: Which of the following statement are true? (Check all that apply)

**Answers**:

1. "The inception score only requires generated images"

- Answer: False
- Explanation: Inception Score requires both generated images and a pre-trained Inception model to calculate the
  conditional and marginal distributions.

2. "The inception score requires to calculate the KL divergence between the conditional label distribution and the
   marginal distribution"

- Answer: True
- Explanation: The Inception Score calculates the KL divergence between p(y|x) (conditional) and p(y) (marginal)
  distributions.

3. "The Frechet Inception Distance requires both the mean and covariance of the generated samples and the real samples"

- Answer: True
- Explanation: FID uses both statistical measures (mean and covariance) from both real and generated samples to compare
  distributions.

4. "The Frechet Inception Distance calculates the distance between two multivariate Gaussian distributions"

- Answer: True
- Explanation: FID measures the distance between two multivariate Gaussian distributions fitted to the real and
  generated data features.

Task 2:
Correction: At 1:10, the description of the $m_f$ and $C_f$ should be

- $m_f$: mean of the fake distribution
- $C_f$: covariance of the fake distribution

Frechet Inception Distance or FID measures the distance between two multinomial Gaussian distributions, where the mean
and covariance are calculated from the real and the generated samples.


<br>

![localImage](/images/frechet.png)

<br>

The mathematical equation for determining FID is:

$d = ||m_r - m_f||_2^2 + \text{Tr}(C_r + C_f - 2(C_rC_f)^{1/2})$

where:

- $m_r$: mean of the real distribution
- $C_r$: covariance of the real distribution
- $m_f$: mean of the fake distribution
- $C_f$: covariance of the fake distribution
- Tr: trace

The Inception Score paper and the Frechet Inception Distance paper (which is also the TTUR paper, suprise!) contain a
lot more information about the implementation of both metrics.

Both official implementations are available:

- Inception Score Code
- Frechet Inception Distance code

### Other Applications of GANs

### Semi-Supervised Learning

<br>

Semi-supervised models are used when you only have a few labeled data points. The motivation for this kind of model is
that, we increasingly have a lot of raw data, and the task of labelling data is tedious, time-consuming, and often,
sensitive to human error. Semi-supervised models give us a way to learn from a large set of data with only a few labels,
and they perform surprisingly well even though the amount of labeled data you have is relatively tiny. Ian Goodfellow
has put together a video on this top, which you can see, below.


<br>

![localImage](/images/semi_supervised.png)

<br>


Semi-Supervised Learning in PyTorch
There is a readable implementation of a semi-supervised GAN from the repository Improved GAN (Semi-supervised GAN)(opens
in a new tab). If you'd like to implement this in code, I suggest reading through that code!

Domain Invariance
Consider the car classification example from the research paper on arXiv(opens in a new tab). From the abstract,
researchers (Timnit Gebru, et. al) wanted to:

develop a computer vision pipeline to predict income, per capita carbon emission, crime rates and other city attributes
from a single source of publicly available visual data. We first detect cars in 50 million images across 200 of the
largest US cities and train a model to predict demographic attributes using the detected cars. To facilitate our work,
we have collected the largest and most challenging fine-grained dataset reported to date consisting of over 2600 classes
of cars comprised of images from Google Street View and other web sources, classified by car experts to account for even
the most subtle of visual differences.

One interesting thing to note is that these researchers obtained some manually-labeled Streetview data and data from
other sources. I'll call these image sources, domains. So Streetview is a domain and another source, say cars.com is
separate domain.

The researchers then had to find a way to combine what they learned from these multiple sources! They did this with the
use of multiple classifiers; adversarial networks that do not include a Generator, just two classifiers.

One classifier is learning to recognize car types
And another is learning to classify whether a car image came from Google Streetview or cars.com, given the extracted
features from that image
So, the first classier’s job is to classify the car image correctly and to trick the second classifier so that the
second classifier cannot tell whether the extracted image features indicate an image from the Streetview or cars.com
domain!

The idea is: if the second classifier cannot tell which domain the features are from, then this indicates that these
features are shared among the two domains, and you’ve found features that are domain-invariant.

Domain-invariance can be applied to a number of applications in which you want to find features that are invariant
between two different domains. These can be image domains or domains based on different population demographics and so
on. This is also sometimes referred to as adversarial feature learning(opens in a new tab).

Ethical and Artistic Applications: Further Reading
Ethical implications of GANs(opens in a new tab) and when "fake" images can give us information about reality.
Do Androids Dream in Balenciaga?(opens in a new tab) note that the author briefly talks about generative models having
artistic potential rather than ethical implications, but the two go hand in hand. The generator, in this case, will
recreate what it sees on the fashion runway; typically thin, white bodies that do not represent the diversity of people
in the world (or even the diversity of people who buy Balenciaga).
GANs for Illuminating Model Weaknesses
GANs are not only used for image generation, they are also used to find weaknesses in existing, trained models. The
adversarial examples that a generator learns to make, can be designed to trick a pre-trained model. Essentially, small
perturbations in images can cause a classifier (like AlexNet or a known image classifier) to fail pretty spectacularly!

Adding a small amount of noise to an image of a panda causes a model to misclassify it as a gibbon(opens in a new tab),
which is a kind of ape. One of the interesting parts of this is the model's confidence. With this noise it is 99.3%
confident that this is an image of a gibbon, when we can pretty clearly see that it is a panda!

# Image to Image Translation

Image to Image Translation means using GANs to map from one type of image to another type, to create a new image.

This lesson will focus on a particular image-to-image translation architecture, known as the CycleGAN model.

<br>

![localImage](/images/cycle.png)

<br>

By the end of this lesson, you will be able to:

1. Implement unpaired images dataloaders
2. Build residual blocks and incorporate them in the CycleGAN generator
3. Train a CycleGAN model on an image dataset

In this lesson on Image to Image Translation, you will:

1. Build Unpaired Images Dataloader
2. Build the CycleGAN Generator
3. Implement the CycleGAN Loss Function
4. Train CycleGAN model

Generating new data is a challenging task; however, GAN models can learn something about the underlying structure of
training data, to discern patterns that can be used to recreate images.

GANs can also be applied to Image to Image Translation.

Image to Image Translation – takes an input image and produces a transformed image as output

### Applications for Image to Image Translation

Deep learning and computer vision applications of image to image translation include:

1. Semantic Segmentation - every pixel in the input image is labeled and classified
2. Translating an image into a new domain with a desired property or feature

Pix2Pix and CycleGAN are two formulations of image to image translation that learn to transform an input image into a
desired output and they can be applied to a variety of tasks.

# Objective Loss Functions

An objective function is typically a loss function that you seek to minimize (or in some cases maximize) during training
a neural network. These are often expressed as a function that measures the difference between a prediction $\hat{y}$
and a true target $y$:

$\mathcal{L}(y,\hat{y})$.

The objective function we've used the most in this program is cross entropy loss, which is a negative log loss applied
to the output of a softmax layer. For a binary classification problem, as in real or fake image data, we can calculate
the binary cross entropy loss as:

$-[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$

In other words, a sum of two log losses!

Latent Space
Latent means "hidden" or "concealed". In the context of neural networks, a latent space often means a feature space, and
a latent vector is just a compressed, feature-level representation of an image!

For example, when you created a simple autoencoder, the outputs that connected the encoder and decoder portion of a
network made up a compressed representation that could also be referred to as a latent vector.

You can read more about latent space in [this blog post] as well as an interesting property of this space: recall that
we can mathematically operate on vectors in vector space and with latent vectors, we can perform a kind of feature-level
transformation on an image!

This manipulation of latent space has even been used to create an interactive GAN, iGAN(opens in a new tab) for
interactive image generation! I recommend reading the Interactive Image Generation with Generative Adversarial Networks
research paper on ArXiv.

# Pix2Pix Image-to-Image Translation

## 1. Introduction

Pix2Pix is a conditional GAN (cGAN) framework for image-to-image translation tasks like:

- Sketches to photos
- Maps to satellite images
- Black & white to color
- Day to night scenes
- Edges to full images

## 2. Generator Architecture (U-Net)

### Structure:

- **Encoder-Decoder with Skip Connections**

  ```textmate
  Input Image → Encoder → Latent Space → Decoder → Output Image
        ↓         ↓                        ↑
        Skip Connections ------------------↑
  ```

### Key Components:

1. **Encoder**:
    - Downsampling blocks
    - Each block: Conv2D → BatchNorm → LeakyReLU
    - Progressively reduces spatial dimensions

2. **Decoder**:
    - Upsampling blocks
    - Each block: TransposeConv2D → BatchNorm → ReLU
    - Gradually recovers spatial dimensions

3. **Skip Connections**:
    - Connects encoder layers to decoder layers
    - Helps preserve fine details and spatial information
    - Combats information loss in bottleneck

## 3. Discriminator Architecture (PatchGAN)

### Design:

- Focuses on local image patches rather than entire image
- Classifies if each N×N patch is real or fake
- More efficient than full-image discrimination

### Structure:

```textmate
Input: Concatenated (Input Image, Output Image)
↓
Convolutional Layers (No pooling)
↓
Output: Grid of Real/Fake Predictions
```

### Features:

- Smaller receptive field
- Fewer parameters than full-image discriminator
- Better at capturing high-frequency details
- Penalizes structure at patch scale

## 4. Loss Functions

### Generator Loss:

1. **Adversarial Loss**:
    - Fool the discriminator
    - Make generated images look realistic

2. **L1 Loss**:
    - Pixel-wise difference between generated and target
    - Encourages output to match ground truth

Combined Loss = λ₁(Adversarial Loss) + λ₂(L1 Loss)

### Discriminator Loss:

- Standard GAN loss
- Binary cross-entropy for real/fake classification

## 5. Training Process

1. **Input Preparation**:
    - Paired images (source → target)
    - Normalize to [-1, 1]
    - Data augmentation if needed

2. **Training Steps**:

```textmate
For each batch:
   1. Generate fake images using G
   2. Update D using real and fake images
   3. Update G using combined loss
   4. Apply gradient penalties if needed
```

<br>

![localImage](/images/Pix2Pix.png)

<br>

## 6. Key Features & Improvements

1. **Conditional Input**:
    - Generator sees source image
    - Discriminator sees both source and output

2. **Noise Handling**:
    - Dropout in generator provides noise
    - Used during both training and testing

3. **Architecture Choices**:
    - No pooling layers
    - Instance normalization
    - Appropriate padding for size preservation

## 7. Common Applications

- Photo colorization
- Facade generation
- Street scene rendering
- Medical image synthesis
- Style transfer

## 8. Limitations

- Requires paired training data
- Mode collapse possible
- Limited to learned transformations
- Resolution constraints
- Training stability issues

This framework provides a powerful approach for supervised image-to-image translation tasks, with the flexibility to be
adapted for various applications.

Pix2Pix Discriminator

**Quiz Question**: Match each component of the Pix2Pix model with its function

**Answers**:

The Pix2Pix model required a paired dataloader that outputs the same observation in both domain. Using an
encoder-decoder generator and a discriminator, we can learn the mapping from one domain to another.

**Quiz Question**: Match each component of the Pix2Pix model with its function

| Component                 | Function                                                                         | Explanation                                                                                                               |
|---------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Pix2Pix Generator         | has an encoder - decoder architecture                                            | - Uses U-Net architecture<br>- Contains skip connections<br>- Transforms input images between domains                     |
| Pix2Pix paired dataloader | outputs the same observation in both domains                                     | - Manages paired training data<br>- Ensures corresponding images from source/target domains<br>- Maintains data alignment |
| Pix2Pix discriminator     | - classifies images between the two domains<br>- classifies real from fake image | - Uses PatchGAN architecture<br>- Works on image patches<br>- Provides feedback on translation realism                    |

The table format makes it clearer to:

- Match components with functions
- See relationships between elements
- Understand the role of each part
- Get detailed explanations in an organized way

This format is especially useful for matching-type questions as it clearly shows the connections between different
elements.

1. **Pix2Pix Generator**:

- Function: "has an encoder - decoder architecture"
- Explanation: Uses U-Net architecture with encoder-decoder structure and skip connections to transform images between
  domains

2. **Pix2Pix paired dataloader**:

- Function: "outputs a the same observation in both domains"
- Explanation: Loads paired training data where each sample has corresponding images from both source and target domains

3. **Pix2Pix discriminator**:

- Functions:
    - "classifies images between the two domains"
    - "classifies real from fake image"
- Explanation: PatchGAN discriminator that:
    - Determines if image pairs are real or generated
    - Assesses if translations between domains are realistic
    - Works on patches rather than whole images

Key Points:

- Generator transforms while preserving structure
- Dataloader ensures paired training data
- Discriminator provides adversarial feedback for realistic translations

This setup enables supervised image-to-image translation by maintaining correspondence between domains through paired
data.

### CycleGANs & Unpaired Data

In practice, paired data is time-intensive and difficult to collect. In some cases, such as stylized images, paired data
is impossible to get.

With unpaired data, there is no longer the ability to look at real and fake pairs of data - but the model can be changed
to produce an output that belongs to the target domain.

Cycle Consistency Constraint uses inverse mapping to accomplish this task
Many of the images in the video above are collected in the Pix2Pix and CycleGAN Github repository(opens in a new tab)
developed by Jun-Yan.

### Cycle Consistency Loss

Importance of Cycle Consistency
A really interesting place to check cycle consistency is in language translation. Ideally, when you translate one word
or phrase from, say, English to Spanish, if you translate it back (from Spanish to English) you will get the same thing!

In fact, if you are interested in natural language processing, I suggest you look into this as an area of research; even
Google Translate has a tough time with this. In fact, as an exercise, I want you to see if Google Translate passes the
following cycle consistency test.

Model Shortcomings
As with any new formulation, it's important not only to learn about its strengths and capabilities but also, its
weaknesses. A CycleGAN has a few shortcomings:

1. It will only show one version of a transformed output even if there are multiple, possible outputs.
2. A simple CycleGAN produces low-resolution images, though there is some research around high-resolution GANs(opens in
   a new tab)
3. It occasionally fails!

### When to Use Image to Image Translation

One of the challenges with deep learning based solutions is the amount of required data. It takes a significant amount
effort and money to:

Capture real data
Clean real data
Annotate real data
Alternative Data Sources
Another source for data is computer generated data or synthetic data. Synthetic data can be used to train models on
tasks such as object detection or classification.

However, because synthetic images are still quite different from real images and model performance is usually not on par
with models trained on real data.

The difference between the real and the synthetic domain is called domain gap.

# Introduction to Modern GANs

In this lesson, we will cover how the GAN architectural paradigm has been rethought over the last few years. We will
cover topics such as the:

1. Wasserstein GAN architecture
2. Gradients to improve GAN training stability
3. Growing architectures generators and discriminators
4. StyleGAN model

<br>

![localImage](/images/modern_gan.png)

<br>

In this lesson on Modern GANs, you will:

1. Use the Wasserstein Distance as a Loss Function for Training GANs
2. Leverage Gradient Penalties to Stabilize GAN Model Training
3. Build a ProGAN Model
4. Build Components of a StyleGAN Model

The original GAN paper [1] already mentions some of the limitations of the BCE Loss, in the section 6 'Advantages and
disadvantages'.

The MiniMax game

This is the minimax game that you should be familiar with.

$E[\log(D(x))] + E[\log(1-D(G(z)))]$

- We have $x$, a sample from our real distribution, $z$ the latent vector, our discriminator $D$, and our generator $G$.

- The discriminator tries to maximize this expression, which means maximizing the log probability of $x$ being real and
  maximizing the log of the inverse probability of $G(z)$ being real.

- The generator tries to minimize the log of the inverse probability of $G(z)$ being real.

- It is more stable for the generator to maximize the log probability of $G(z)$ being fake.

Challenges of Training GANs
The common problems with GANs are:

1. Mode Collapse occurs when the generator only creates some of the modes of the real distribution.
2. Vanishing Gradient occurs when the discriminator loss reaches zero and the generator is not learning anymore.

Addressing Vanishing Gradients
Least squares (LSGANs) can partly address the vanishing gradient problem for training deep GANs.

The problem is as follows:

For negative log-likelihood loss, when an input x is quite big, the gradient can get close to zero and become
meaningless for training purposes. However, with a squared loss term, the gradient will actually increase with a larger
x, as shown below.

Least square loss is just one variant of a GAN loss. There are many more variants such as a Wasserstein GAN loss(opens
in a new tab) [3] and others.

These loss variants sometimes can help stabilize training and produce better results. As you write your own code, you're
encouraged to hypothesize, try out different loss functions, and see which works best in your case!

### Wasserstein Loss

Here's the OCR text with LaTeX equations:

To prevent mode collapse and vanishing gradient there is another loss function to train GANs:

- The Earth Mover Distance or Wasserstein Metric also referred to as Wasserstein Loss and Wasserstein Distance

The Wasserstein Loss is mathematically represented as follows:

$E[C(x)] - E[C(G(z))]$

Similar to the BCE Loss, note that the logs have disappeared. Indeed the Wasserstein distance gets rid of the log
function and only considers the probabilities.

With the Wasserstein distance the discriminator is called Critic.

The Critic:

Does not discriminate between real and fake anymore but instead measures the distance between both distributions.
Will try to maximize this expression.
Wants to maximize the score of the real distribution and minimize the score of the fake distribution, which is similar
to maximizing its inverse.
The generator will try to maximize the critic score of the fake distribution, which is similar to minimizing it with the
flipped label.

The WGAN minimax game is described by the formula above.

When training the critic $C$, we want to maximize the critic score on the real images x and minimize the critic score on
the fake images $G(z)$ which is similar to maximizing the inverse of $C(G(z))$.

When training the generator $G$, we want to maximize the score of the critic for the fake images.

# 1-Lipschitz continuous

The 1-Lipschitz continuous is a new constraint on the discriminator or critic when using the Wasserstein distance as a
loss function.

Defined mathematically:

For a function $f$, the following condition must be fulfilled: $|\frac{df(x)}{dx}| \leq 1$

Note: For the rest of the class, Critic and Discriminator will be used interchangeably to designate the Discriminator
network.

**Quiz Question**: Which ones of the following are true about the BCE Loss?

| Statement                                                                             | Answer | Explanation                                                                                                                                              |
|---------------------------------------------------------------------------------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| The output of the discriminator is bounded between 0 and 1                            | TRUE   | - Uses sigmoid activation for binary classification<br>- Outputs represent probability distribution<br>- Essential for binary cross-entropy calculations |
| Requires the discriminator to be 1-Lipschitz continuous                               | FALSE  | - This is a requirement for Wasserstein GANs (WGAN)<br>- Not needed for standard BCE loss<br>- BCE uses different constraints                            |
| Can lead to vanishing gradients when the real and fake distributions are very similar | TRUE   | - When distributions overlap significantly<br>- Gradient information becomes too small<br>- Makes training unstable                                      |
| Can lead to mode collapse                                                             | TRUE   | - Generator might focus on a single mode to fool discriminator<br>- Loses diversity in generated samples<br>- Common problem with BCE loss               |

**Key Points**:

- BCE Loss has both mathematical limitations (vanishing gradients) and training issues (mode collapse)
- Unlike Wasserstein loss, doesn't require Lipschitz continuity
- Output constraints are fundamental to how BCE works
- These limitations led to development of alternative losses like Wasserstein

WGAN Training Algorithm
To train the GAN:

Sample from the real data and generate some fake samples
Then calculate the Wasserstein Distance and the gradients at line 5.
Line 6, we perform backpropagation using RMSprop, an optimization algorithm similar to stochastic gradient descent or
Adam
Enforce the weights to stay within a threshold of -0.01 and 0.01

# WGAN algorithm

1 while GAN has not converged:
2 # for each iteration
3 Sample a batch of real data
4 Sample a batch of fake data
5 Calculate the gradient gw
6 Backpropagation w ← w + ɑ RMSProp
7 w ← clip(w, -c, c)
In short, weight clipping works, but is not the best approach to enforce the Lipschitz constraint.

# Gradient Penalties

The WGAN-GP paper introduced the concept of gradient penalty. In this paper, the authors add a new term to the new loss
function that will penalize high gradients and help enforce the1-Lipschitz constraint. Added to the Wasserstein Loss
formula was the gradient penalty:

$\lambda E[(||\nabla_{\hat{x}}C(\hat{x})||_2 - 1)^2]$

$E[C(X)] - E[C(G(z))] + \lambda E[(||\nabla_{\hat{x}}C(\hat{x})||_2 - 1)^2]$

Wasserstein Loss Gradient penalty

$\lambda$: penalty coefficient
$\hat{x}$: uniformly sampled data point between the real and fake distributions

Mathematical representation of gradient penalty

Calculating the gradient penalty also includes a bit of interpolation of the real and fake distribution:

1. Randomly sample a coefficient $\alpha$ of between 0 and 1
2. Calculate the interpolation as: $\hat{x} = \alpha x + (1-\alpha)G(z)$

# Gradient Penalties

The WGAN-GP paper introduced the concept of gradient penalty. In this paper, the authors add a new term to the new loss
function that will penalize high gradients and help enforce the1-Lipschitz constraint. Added to the Wasserstein Loss
formula was the gradient penalty:

$\lambda E[(||\nabla_{\hat{x}}C(\hat{x})||_2 - 1)^2]$

$E[C(X)] - E[C(G(z))] + \lambda E[(||\nabla_{\hat{x}}C(\hat{x})||_2 - 1)^2]$

Wasserstein Loss Gradient penalty

$\lambda$: penalty coefficient
$\hat{x}$: uniformly sampled data point between the real and fake distributions

Mathematical representation of gradient penalty

Calculating the gradient penalty also includes a bit of interpolation of the real and fake distribution:

1. Randomly sample a coefficient $\alpha$ of between 0 and 1
2. Calculate the interpolation as: $\hat{x} = \alpha x + (1-\alpha)G(z)$

### Progressive Growing of GANS

To make training even more stable, the ProGAN model was developed and the current resolution is 16x16.

How ProGAN works

1. It adds a new layer to the generator and a new layer to the discriminator by fading the layers in smoothly.
2. In the generator, the resolution of the 16x16 layer is doubled using an interpolation method such as nearest
   neighbor. The output of the 32x32 layer is then fused with this interpolated output.
3. In the discriminator, the output of the 32x32 layer is fused with a downsampled image.
4. A pooling method such as average pooling is used for downsampling.

<br>

![localImage](/images/progan.png)

<br>


In both cases, perform a weighted sum of the learned output of the new layer with the non-parametric output of the
previous layer.

Slowly increase the weight of the output of the new layer over 10 epochs to reach a stage where there is no need to fade
that layer anymore.

Then train the network at the 32x32 resolution for another 10 epochs.

Layer Fading
For more stable training, layer fading is a way to incorporate new layers. Consider the following example:

<br>

![localImage](/images/layer.png)

<br>

1. Training a ProGAN model and the current resolution is 16x16. The toRGB layer maps the output of the last convolution
   to an RGB image and the from RGB layer takes a RGB image as input and feeds it into the next convolution layer.
2. To increase the resolution to 32x32 use layer fading. Add a new layer to the generator and the discriminator by
   doubling the resolution of the 16x16 layer using an interpolation method such as nearest neighbor.
3. In the generator, fuse the output of the 32x32 layer with the interpolated output.
4. In the discriminator, fuse the output of the 32x32 layer with a downsampled image and use a pooling method such as
   average pooling for downsampling.
5. For both cases, perform a weighted sum of the learned output of the new layer with the non parametric output of the
   previous layer. Slowly increase the weight of the output of the new layer over 10 epochs to reach a stage where a
   fade is not needed in that layer.
6. Train the network at the 32x32 resolution for another 10 epochs

ProGAN Tricks

1. Progressive Growing – Progressively train layers and increase resolution
2. Minibatch Discrimination – Enforce fake and real batches to have similar statistics
3. Equalized Learning Rates – Scale the weights of each layer by a different constant to make sure the layers are
   learning at the same speed
4. Pixelwise Normalization – Normalize each pixel of a feature map along the channel axis

Pixelwise normalization
You are familiar with batch normalization and you may be familiar with other type of normalization, as described in the
figure below.



<br>

![localImage](/images/group_normalization.png)

<br>


C is the channel dimensions, N the batch dimension and H, W the spatial dimensions. For example, for a batch
normalization layer, we calculate mean and variance over the batch and spatial dimensions, so we have a pair of (mean,
variance) values for each channel.

With pixel normalization, we normalize each pixel of the input volume as follow:

y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()
x = x / y
where x is the input volume of dimensions (NCHW). We square the input volume, calculate the mean over the channel
dimension, add a very small factor alpha and calculate the square root.

Minibatch Standard Deviation
The paper, Improved Techniques for Training GANs(opens in a new tab) [3], introduced the concept of minibatch
discrimination, to enforce similarities between batches of real and fake images.

In the ProGAN paper, the authors simplify this idea by introducing minibatch standard deviation. They create a new layer
that adds a feature map to the input. This layer does the following:

calculate the standard deviation for each feature and spatials locations
replicate the value and concatenate it over all spatial locations

```textmate
def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.
```

The code above is taken from the original implementation in TensorFlow(opens in a new tab) but the PyTorch version is
very similar.

Note how the authors are calculating the standard deviation per group of 4 pixels here.

This new layer is added only in the discriminator obviously and towards the end of the network.

### StyleGAN: Introduction

Deep learning is a somewhat recent field and many consider the 2012 AlexNet paper as the starting point of the deep
learning revolution. The progress in creating realistic generated images is most exemplified by the StyleGAN paper in
2019 as it was the first architecture to produce very high-quality samples.

The Traditional Generator
For a traditional generator:

We input a latent vector z.
Run it through a bunch of fully connected, convolution and normalization layers.
Get a generated RGB image.
The StyleGAN Generator
For the StyleGAN generator :

There is a new network, only made of fully connected layer, the mapping network, and it is taking the latent vector and
outputs a new latent vector w.
Add noise at multiple places in the network, always after the convolution layers.
StyleGAN uses a new type of normalization layer, the adaptive instance normalization layer, or AdaIn.
Next, we will dissect each one of these new components and understand how they were leveraged to create such high
quality images.

<br>

![localImage](/images/style_gan.png)

<br>

### StyleGAN Components

**StyleGAN in Simple Terms:**
Think of StyleGAN as an AI artist that can:

- Create images with specific styles (like painting portraits)
- Control different aspects separately (like hair color, age, facial features)
- Add realistic details (like skin texture, wrinkles)
- Mix different styles together (like combining two faces)

**StyleGAN Technical Definition:**
A generative adversarial network architecture that:

- Separates high-level attributes through style mapping
- Uses adaptive normalization for feature manipulation
- Implements stochastic variation for detail generation
- Enables disentangled style control at different scales

**StyleGAN Components Table:**

| Component                               | Function                                                                                    | Technical Details                                                                                                  | Real-World Analogy                                                                 |
|-----------------------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Mapping Network                         | Fully connected layers that map latent vector z to latent vector w. Helps with entanglement | - Transforms random noise (z) into style vectors (w)<br>- Multiple FC layers<br>- Improves feature disentanglement | Like a translator converting raw ideas into specific style instructions            |
| Noise Injection                         | Adding Gaussian noise at multiple places in the generator helps with stochastic variation   | - Adds random noise at different resolutions<br>- Controls fine detail generation<br>- Creates natural variations  | Like adding random texture details to make images more realistic (pores, wrinkles) |
| AdaIN (Adaptive Instance Normalization) | Projects the latent vector w to styles and injects them into the generator                  | - Normalizes feature maps<br>- Applies style-based transformations<br>- Controls specific attributes               | Like an artist's tool that applies specific styles to different parts of the image |

**Key Features:**

1. **Progressive Generation:**
    - Builds images from low to high resolution
    - Maintains consistency across scales

2. **Style Control:**
    - Separate control over different features
    - Ability to mix styles at different levels

3. **Quality Improvements:**
    - Better feature disentanglement
    - More realistic detail generation
    - Improved style control

This architecture revolutionized GAN-based image generation by providing better control and quality in generated images.

Controllable Generation
Conditional Generation indicates that the training set must be labeled and conditioning is limited to examples from the
training set.

Conditional Generation – each image is conditioned with a label (e.g. MNIST dataset).
For example, fake 11s can not be generated with a conditional GAN trained on the MNIST dataset because the data set only
includes digits in the range of 0 to 9.

Conditional Generative Adversarial Nets(opens in a new tab) [1] introduced the conditional GAN. An example of
conditional implementation in PyTorch can be viewed in the PyTorch-GAN CGAN implementation on GitHub(opens in a new
tab). Conditional GANs have an extra input from the discriminator and generator networks.

For controllable generation, instead of inputting a label to condition the output, the latent vector z is modified to
control the aspect of the output. This makes the assumption that the components of z each control a different aspect of
the output image.

Controllable Generation – does not require labels.
The Mapping Network
The mapping network is a new component of the StyleGAN generator. A mapping network:

1. Takes the latent vector z as input
2. Outputs a new latent vector w
3. Helps to disentangle the latent vector z for controllable generation.

<br>

![localImage](/images/mapping_network.png)

<br>


The Entanglement Problem
When modifying some components, we impact more than one feature. This is the entanglement problem.

For example in trying to generate faces, features could include:

1. Haircut
2. Eye color
3. Glasses
4. Age

<br>

![localImage](/images/entanglement.png)

<br>


If the features are entangled, putting glasses on a person could also make them older.

Mapping network to the rescue! By mapping the vector z to another vector w, the generator gets the capacity to
disentangle features.

Noise Injection
Another new component of StyleGAN is the injection of noise at different locations in the generator. This noise
injection will:

Help with stochastic variation! Injecting noise in the network will help create more diverse features.
Happen at different locations in the network and impacts the variability of the images at different levels.
To add noise:

1. A random feature map is sampled from a gaussian distribution
2. The map is multiplied by a learned scaling factor
3. This noise is applied to the output of the convolutional layers

<br>

![localImage](/images/noise_injection.png)

<br>


All Normalization Layers calculate the mean and the variance of a certain subset and normalize the input.

Remember, for Batch Normalization, we:

1. Calculate the mean and variance of the batch and spatial dimensions
2. For each channel of the inputs, there are different values of means and variance

Instance Normalization Layer
The Instance Normalization Layer – only normalizes over the spatial dimensions and each input has a number of channels
times the batch size values of means and variance.



<br>

![localImage](/images/norm_layer.png)

<br>

Adaptive Instance Normalization Layer
The Adaptive Instance Normalization Layer (Adaln):

1. Takes the latent vector, w, as input and using a fully connected layer, projects that vector into two vectors of
   style, $y_s^y_s^$ and $y_b^y_b^$
2. The output of the previous layer goes through an Instance Normalization Layer.
3. Use the styles $y_s^y_s^$ and $y_b^y_b^$ to scale and bias the output of the Instance Normalization Layer.
4. Allows one to project the latent vector w into the styles and inject the styles into the generator.

Style Mixing injects a different vector w at different places in the network and provides a regularization effect. This
prevents the network from assuming that adjacent styles are correlated.


<br>

![localImage](/images/adaptive.png)

<br>

Style Transfer
In practice, Adaln layers allow for the creation of a new image (c) by taking a first image (a) and modifying it in the
style of a second image (b). A popular example is taking the image of the Mona Lisa (a) and the style of a Picasso
painting (b) and creating a new Mona Lisa in Picasso style image (c). This image can be seen here(opens in a new tab)
and this process is known as style transfer.

The initial process of style transfer was time consuming; however, check out the paper, Arbitrary Style Transfer in
Real-time with Adaptive Instance Normalization(opens in a new tab), which details a use case for how Adaln layers may be
implemented to create fast style transfers based on arbitrary styles.

### When to Use Modern GAN Techniques

Starting with a simpler architecture is always an easy and fast way to get started on a new problem.

DCGAN is great starting point
ProGAN or StyleGAN are practical when training on high resolution images
Wasserstein Loss and Gradient Penalties experimentation is recommended when mode collapse or vanishing gradient are
observed

––––––––––––––––––––––––––––––––––––––––––––––

<br>

![localImage](/images/adaptive.png)

<br>
––––––––––––––––––––––––––––––––––––––––––––––

## GANs Project: Face Generation

### Getting the project files

The project files are located in the Project Workspace and include the following files:

* **`dlnd_face_generation_starter.ipynb`**
* **`README.md`**
* **`requirements.txt`**
* **`tests.py`**
* **`processed-celeba-small.zip`**

We highly recommend using the Project Workspace to complete your project; however, if you choose to not use the
workspace, you can download the project files from the Project Workspace.

### Instructions

Open the notebook file, `dlnd_face_generation_starter.ipynb` and follow the instructions. This project is organized as
follows:

* **Data Pipeline**: implement a data augmentation function and a custom dataset class to load the images and transform
  them.
* **Model Implementation**: build a custom generator and a custom discriminator to make your GAN
* **Loss Functions and Gradient Penalty**: decide on loss functions and whether you want to use gradient penalty or not.
* **Training Loop**: implement the training loop and decide on which strategy to use

Each section requires you to make design decisions based on the experience you have gathered in this course. Do not
hesitate to come back to a section to improve your model or your data pipeline based on the results that you are
getting.

Building a deep learning model is an iterative process, and it's especially true for GANs! Good luck!

### Submitting Your Project

For this project you will need to submit one file – **dlnd_face_generation.ipynb**

The full project may be submitted in two ways:

**Project completed in Project Workspace:**

* Your project may be submitted directly via the Project Workspace by pressing the **`Submit`** button in the bottom
  right corner of the workspace.

**Project completed outside of Project Workspace:**

* Your project may be submitted using the Project Submission page by pressing the **`Submit Project`** button in the top
  right corner of the page and following those directions.
* You will need to create a zip file of the required project file and submit the zip file.
