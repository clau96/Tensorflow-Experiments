import tensorflow as tf
from tensorflow.keras import layers, optimizers, metrics, Sequential, Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy

class vanillaGAN(Model):
    """
    Implementation of a GAN using Tensorflow 2.0
    """
    def __init__(self, hlayers, img_dims, noise_dim=100, learning_rate=2e-4, beta_1=0.5, beta_2=0.999, slope=0.2, batch_size=64, epochs=100):

        super().__init__()

        """
        Initialise the hyperparameters
        """
        self.hlayers = hlayers
        self.noise_dim = noise_dim
        self.slope = slope
        self.img_dims = img_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate

        self.generator = self.generator(training=True)
        self.g_optimizer = optimizers.Adam(lr=self.lr, beta_1=beta_1, beta_2=beta_2)

        self.discriminator = self.discriminator(training=True)
        self.d_optimizer = optimizers.Adam(lr=self.lr, beta_1=beta_1, beta_2=beta_2)

    def generator(self, training=False):
        """
        Defines the generator of the model.
        """
        generator = Sequential()

        generator.add(layers.Input(shape=(self.noise_dim,)))

        generator.add(layers.Dense(self.hlayers["G"][0], activation=LeakyReLU(self.slope)))
        generator.add(layers.BatchNormalization(trainable=training))

        generator.add(layers.Dense(self.hlayers["G"][1], activation=LeakyReLU(self.slope)))
        generator.add(layers.BatchNormalization(trainable=training))

        generator.add(layers.Dense(self.hlayers["G"][2], activation=LeakyReLU(self.slope)))
        generator.add(layers.BatchNormalization(trainable=training))

        generator.add(layers.Dense(self.img_dims[0]*self.img_dims[1]*self.img_dims[2], activation='tanh'))

        return generator

    def discriminator(self, training=False):
        """
        Defines the discriminator of the model.
        """

        discriminator = Sequential()

        discriminator.add(layers.Flatten(input_shape=(self.img_dims)))

        discriminator.add(layers.Dense(self.hlayers["D"][0], activation=LeakyReLU(self.slope)))
        discriminator.add(layers.BatchNormalization(trainable=training))

        discriminator.add(layers.Dense(self.hlayers["D"][1], activation=LeakyReLU(self.slope)))
        discriminator.add(layers.BatchNormalization(trainable=training))

        discriminator.add(layers.Dense(self.hlayers["D"][2], activation=LeakyReLU(self.slope)))
        discriminator.add(layers.BatchNormalization(trainable=training))

        discriminator.add(layers.Dense(units=1, activation=tf.nn.sigmoid))

        return discriminator

    def loss_fn(self, labels, logits):
        
        bce = BinaryCrossentropy(from_logits=True)
        loss = tf.reduce_mean(bce(labels, logits))
        
        return loss

    def train_step(self, real_images):
        """
        One training step of the GAN algorithm.

        In the paper they recommend maximising Log(D(G(z))) instead of minimizing Log(1-D(G(Z))) to avoid vanishing gradients.
        -> We want the generator to maximise the number of incorrect predictions made by the discriminator, instead of minimise the number of correct predictions.
        """
        
        batch_size = tf.shape(real_images)[0]

        # Sample random point in latent space
        noise = tf.random.normal(shape=[batch_size, self.noise_dim])

        # Decode noise to fake images
        generated_images = self.generator(noise, training=True)


        
        # Combine them with real images
        combined_images = tf.concat([generated_images, tf.reshape(real_images, (real_images.shape[0], self.img_dims[0]*self.img_dims[1]*self.img_dims[2]))], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Sample random points in the latent space
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        # Misleading labels
        labels = tf.zeros((batch_size, 1))

        # Train generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(noise, training = True))
            g_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return g_loss, d_loss

    def train(self, x_train):

        training_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(self.batch_size) 
        g_losses, d_losses = [], []

        for epoch in range(self.epochs):

            for batch in training_data:

                g_loss, d_loss = self.train_step(batch)
                g_losses.append(g_loss)
                d_losses.append(d_loss)

            print(f'Epoch {epoch+1} | G Loss: {g_loss:.1f} | D Loss: {d_loss:.1f}')


        return g_losses, d_losses


