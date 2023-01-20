# COMS 4995: Applied Deep Learning Final Project
import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
import time

from IPython import display
from tensorflow.keras import layers

# ========== Hyper-parameters ==========
__author__ = ["Tiancheng (Robert) Shi"]
batch_size = 64
epochs = 100
img_height, img_width = (120, 200)
momentum_batch = 0.9  # Default 0.999
momentum_beta = 0.5
noise_dim = 100
num_examples_to_generate = 1
lr = 1e-4
seed = 3407

tf.random.set_seed(seed)

# ========== Data loading and preprocessing ==========
data_root = pathlib.Path("images")
data_image = tf.keras.utils.image_dataset_from_directory(
    data_root, label_mode = None,
    batch_size = batch_size, image_size = (img_height, img_width)
)

for image_batch in data_image:
    print(image_batch.shape)
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.axis("off")
    plt.show()
    break

normalize_layer = tf.keras.layers.Rescaling(1 / 255)
data_norm = data_image.map(lambda img: normalize_layer(img))


# ========== Model construction ==========
class Discriminator(tf.keras.Model):
    """
    Optimized to differentiate real images in training set from fake images created by Generator.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, (5, 5), strides = 2, padding = "same", input_shape = (img_height, img_width, 3))
        self.conv2 = layers.Conv2D(128, (5, 5), strides = 2, padding = "same")
        self.conv3 = layers.Conv2D(256, (5, 5), strides = 2, padding = "same")
        self.d1 = layers.Dense(256)
        self.out_layer = layers.Dense(1, name = "out")
        self.leaky_relu = layers.LeakyReLU(0.2)
        self.drop = layers.Dropout(0.2)
        self.bn1 = layers.BatchNormalization(momentum = momentum_batch)
        self.bn2 = layers.BatchNormalization(momentum = momentum_batch)
        self.bn3 = layers.BatchNormalization(momentum = momentum_batch)
        self.bn4 = layers.BatchNormalization(momentum = momentum_batch)
        self.flatten = layers.Flatten()

    def __call__(self, x, training = False):
        x = self.leaky_relu(self.conv1(x))
        if training:
            x = self.bn1(x)
            # x = self.drop(x)

        x = self.leaky_relu(self.conv2(x))
        if training:
            x = self.bn2(x)
            # x = self.drop(x)

        x = self.leaky_relu(self.conv3(x))
        if training:
            x = self.bn3(x)
            # x = self.drop(x)

        x = self.flatten(x)
        x = self.leaky_relu(self.d1(x))
        if training:
            x = self.bn4(x)
            x = self.drop(x)

        return self.out_layer(x)


class Generator(tf.keras.Model):
    """
    Optimized to generate fake images that confuse the Discriminator without knowing the training set.
    """
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.in_layer = layers.Dense(
            256, activation = "relu",
            use_bias = False, input_shape = (latent_dim, ), name = "in"
        )
        self.d1 = layers.Dense(img_height * img_width * 8, activation = "relu", use_bias = False)
        self.convt1 = layers.Conv2DTranspose(
            256, (5, 5), strides = 1, padding = "same",
            activation = "relu", use_bias = False
        )
        self.convt2 = layers.Conv2DTranspose(
            128, (5, 5), strides = 2, padding = "same",
            activation = "relu", use_bias = False
        )
        self.convt3 = layers.Conv2DTranspose(
            64, (5, 5), strides = 2, padding = "same",
            activation = "relu", use_bias = False
        )
        self.out_layer = layers.Conv2DTranspose(
            3, (5, 5), strides = 2, padding = "same",
            activation = "tanh", use_bias = False
        )
        self.bn1 = layers.BatchNormalization(momentum = momentum_batch)
        self.bn2 = layers.BatchNormalization(momentum = momentum_batch)
        self.bn3 = layers.BatchNormalization(momentum = momentum_batch)
        self.bn4 = layers.BatchNormalization(momentum = momentum_batch)
        self.reshape = layers.Reshape((img_height // 8, img_width // 8, 512))

    def __call__(self, x, training = False):
        x = self.in_layer(x)
        x = self.d1(x)
        if training:
            x = self.bn1(x)

        x = self.reshape(x)
        x = self.convt1(x)
        if training:
            x = self.bn2(x)

        x = self.convt2(x)
        if training:
            x = self.bn3(x)

        x = self.convt3(x)
        if training:
            x = self.bn4(x)

        return self.out_layer(x)


generator = Generator(noise_dim)
discriminator = Discriminator()

# noise_image = generator(tf.random.normal([1, noise_dim]), training = False)
# plt.imshow(noise_image[0, :, :, :], cmap = "gray")
# plt.axis("off")
# plt.show()
# print(discriminator(noise_image))

# ========== Compilation and helper function ==========
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)


def d_loss(data_real, data_fake):
    """
    Discrimination loss function.
    """
    loss_real = cross_entropy_loss(tf.ones_like(data_real), data_real)
    loss_fake = cross_entropy_loss(tf.zeros_like(data_fake), data_fake)
    return loss_real + loss_fake


def g_loss(data_fake):
    """
    Generation loss function.
    """
    return cross_entropy_loss(tf.ones_like(data_fake), data_fake)


d_optimizer = tf.keras.optimizers.Adam(lr, beta_1 = momentum_beta)
g_optimizer = tf.keras.optimizers.Adam(lr, beta_1 = momentum_beta)

# Initialize a random noise base on which fake image is generated.
noise_seed = tf.random.normal([num_examples_to_generate, noise_dim])
subplot_axis = math.ceil(math.sqrt(num_examples_to_generate))


@tf.function
def train_step(images):
    """
    Customized training step.
    Note that real images and fake images should be trained separately.
    """
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated_images = generator(noise, training = True)
        real_output = discriminator(images, training = True)
        fake_output = discriminator(generated_images, training = True)
        loss_gen = g_loss(fake_output)
        loss_disc = d_loss(real_output, fake_output)

    g_grad = g_tape.gradient(loss_gen, generator.trainable_variables)
    d_grad = d_tape.gradient(loss_disc, discriminator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))

    return loss_gen, loss_disc


def generate_and_save_images(model, epoch, test_input):
    """
    Notice `training` is set to False.
    This is so all layers run in inference mode (batchnorm).
    """
    predictions = model(test_input, training = False)
    for i in range(predictions.shape[0]):
        plt.subplot(subplot_axis, subplot_axis, i + 1)
        plt.imshow(predictions[i, :, :, :] / 2 + 0.5)
        plt.axis("off")

    plt.savefig("output/image_at_epoch_{:03d}.png".format(epoch))
    plt.show()


# ========== Training session ==========
avg_gen_loss = []
avg_disc_loss = []
for epoch in range(epochs):
    start = time.time()
    gen_loss = []
    disc_loss = []
    for image_batch in data_norm:
        losses = train_step(image_batch)
        gen_loss.append(losses[0])
        disc_loss.append(losses[1])

    avg_gen_loss.append(np.mean(np.array(gen_loss)))
    avg_disc_loss.append(np.mean(np.array(disc_loss)))

    # Produce images for the GIF as you go
    display.clear_output(wait = True)
    generate_and_save_images(generator, epoch, noise_seed)

    print("Epoch {}\tTime: {}\tGenerator Loss: {}\tDiscriminator Loss: {}".format(
        epoch, time.time() - start, avg_gen_loss[-1], avg_disc_loss[-1]
    ))

# Generate after the final epoch
display.clear_output(wait = True)
generate_and_save_images(generator, epochs, noise_seed)

plt.plot(range(epochs), np.array(avg_gen_loss), label = "Generator")
plt.plot(range(epochs), np.array(avg_disc_loss), label = "Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output/loss.png")
plt.show()
