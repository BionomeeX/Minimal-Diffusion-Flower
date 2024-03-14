import tensorflow as tf


class SPE(tf.keras.layers.Layer):
    def __init__(
        self, output_dims: int = None, channel_index: int = -1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.output_dims = output_dims
        self.channel_index = channel_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dims": self.output_dims,
                "channel_index": self.channel_index,
            }
        )
        return config

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        if len(input_tensor.shape) == 2:
            self.add_intermediate = True

        if self.add_intermediate:
            input_tensor = input_tensor[:, tf.newaxis, ...]

        # Adjust the channel index to positive value if it's negative
        channel_index = (
            self.channel_index
            if self.channel_index >= 0
            else len(input_tensor.shape) + self.channel_index
        )

        # Get the rank of the tensor, excluding batch and channel dimensions
        tensor_rank = len(input_tensor.shape) - 2

        # If output_dims is not specified, derive it based on the input tensor's shape
        if self.output_dims is None:
            tf.assert_equal(
                tf.shape(input_tensor)[-1] % (2 * tensor_rank),
                0,
                f"The specified {self.channel_index} channel dimension {tf.shape(input_tensor)[channel_index]} of the input tensor should be a multiple of 2n with n={tensor_rank}",
            )
            tf.assert_equal(
                tf.shape(input_tensor)[-1] > (2 * tensor_rank),
                True,
                f"The specified {self.channel_index} channel dimension {tf.shape(input_tensor)[channel_index]} of the input tensor should be greater than 2n with n={tensor_rank}",
            )
            output_dim = tf.shape(input_tensor)[channel_index] // (2 * tensor_rank)
        else:
            output_dim = self.output_dims // (2 * tensor_rank)

        # Convert output_dim to float for the following calculations
        output_dim = tf.cast(output_dim, tf.float32)

        # Calculate positional embeddings
        embeddings_log = tf.math.log(10000.0) / (output_dim - 1)
        embeddings = tf.math.exp(
            tf.range(output_dim, dtype=tf.float32) * -embeddings_log
        )

        # Create a meshgrid based on the shape of the input tensor
        positional_axes = [
            tf.range(tf.shape(input_tensor)[i], dtype=tf.float32)
            for i in range(1, tensor_rank + 1)
            if i != channel_index
        ]
        meshgrid = tf.meshgrid(*positional_axes)
        meshgrid = tf.expand_dims(meshgrid, -1)

        # Compute the positional embeddings and apply them to the meshgrid
        embeddings = meshgrid * tf.expand_dims(embeddings, 0)

        # Compute the sinusoidal and cosine embeddings
        sinusoidal_embeddings = [tf.math.sin(embeddings[i]) for i in range(tensor_rank)]
        cos_embeddings = [tf.math.cos(embeddings[i]) for i in range(tensor_rank)]

        # Concatenate the sinusoidal and cosine embeddings
        embeddings = tf.concat(sinusoidal_embeddings + cos_embeddings, axis=-1)

        # Repeat the embeddings along the batch dimension to match the input shape
        embeddings = tf.repeat(
            tf.expand_dims(embeddings, axis=0), tf.shape(input_tensor)[0], axis=0
        )

        # Prepare the transposition vector
        transpose_vector = list(range(tensor_rank + 2))
        if tensor_rank > 1 and channel_index != len(tf.shape(input_tensor)) - 1:
            transpose_vector[1], transpose_vector[channel_index] = channel_index, 1

        # Transpose the embeddings and return
        return tf.transpose(embeddings, transpose_vector)


class Denoiser:
    def __init__(self) -> None:

        n = 8
        input_image = tf.keras.layers.Input((None, None, 3))
        input_time = tf.keras.layers.Input((1,))

        lifted = tf.keras.layers.Conv2D(
            filters=n, kernel_size=(1, 1), strides=(1, 1), padding="same"
        )(input_image)

        time_embed = SPE(output_dims=n)(input_time)
        hidden = tf.keras.layers.Conv2D(
            filters=n, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(lifted)
        hidden = tf.keras.layers.Activation("relu")(hidden)

        hidden += tf.keras.layers.Attention()([hidden, time_embed, time_embed])

        hidden = tf.keras.layers.Conv2D(
            filters=n, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )(hidden)

        hidden = tf.keras.layers.Conv2D(
            filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same"
        )(hidden)
        hidden = tf.keras.layers.Activation("relu")(hidden)
        hidden = tf.keras.layers.LayerNormalization()(hidden)

        self.model = tf.keras.models.Model([input_image, input_time], hidden)

    def __call__(self, image, time):
        return self.model([image[tf.newaxis, ...], tf.Variable(time)[tf.newaxis, ...]])[
            0, ...
        ]

    def save(self, path: str):
        self.model.save(path)

    def load(path: str):
        res = Denoiser()
        res.model = tf.keras.models.load_model(path)
        return res
