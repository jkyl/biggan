import tensorflow as tf
import numpy as np
import glob
import os

def DilatedDenseConv2D(x, hidden_dim=128, kernel_size=3,
                       activation='elu', name=None):
    ''''''
    y = tf.keras.layers.Conv2D(hidden_dim, kernel_size, dilation_rate=2,
                               padding='same', activation=activation,
                               name=name)(x)
    return tf.keras.layers.concatenate([y, x], axis=-1, name=name+'-skip')

def DilatedDenseNet(size, channels=3, kernel_size=3, hidden_dim=128,
                    activation='elu', name=None):
    ''''''
    inp = x = tf.keras.layers.Input((size, size, channels), name=name+'_in')
    depth = int(np.log2(size))
    for i in range(depth):
        x = DilatedDenseConv2D(x, hidden_dim, kernel_size,
                               activation, name=name+'_conv2D-'+str(i))
    out = tf.keras.layers.Conv2D(3, 3, activation=None, padding='same',
                                 name=name+'_out')(x)
    return tf.keras.models.Model(inp, out, name=name)

def BEGAN_decoder(size, channels=3, kernel_size=3,
                  z_dim=128, x_dim=128, activation='elu',
                  n_per_block=2, concat=True, tanh=False, name=None):
    ''''''
    z = x = tf.keras.layers.Input((z_dim,), name=name+'_in')
    x = tf.keras.layers.Dense(8*8*x_dim, name=name+'_dense')(x)
    h0 = x = tf.keras.layers.Reshape((8, 8, x_dim))(x)
    depth = int(np.log2(size // 4))
    for i in range(depth):
        for j in range(n_per_block):
            x = tf.keras.layers.Conv2D(x_dim, kernel_size,
                                activation=activation, padding='same',
                                name=name+'_conv2D-{}.{}'.format(i+1, j+1))(x)
        if i < depth - 1:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                                name=name+'_upsampling-'+str(i))(x)
            if concat:
                h0 = tf.keras.layers.UpSampling2D(size=(2, 2),
                                name=name+'_skip-upsampling-'+str(i))(h0)
                x = tf.keras.layers.concatenate([x, h0], name=name+'-skip'+str(i))

    x = tf.keras.layers.Conv2D(channels, kernel_size, activation=None,
                               padding='same', name=name+'_out')(x)
    out = tf.keras.layers.Activation('tanh')(x) if tanh else x
    return tf.keras.models.Model(inputs=z, outputs=out, name=name)

def BEGAN_encoder(size, channels=3, kernel_size=3,
                  z_dim=128, x_dim=128, activation='elu',
                  n_per_block=2, name=None):
    ''''''
    inp = x = tf.keras.layers.Input((size, size, channels), name=name+'_in')
    depth = int(np.log2(size // 4))
    x = tf.keras.layers.Conv2D(x_dim, kernel_size,
                               padding='same', activation=activation,
                               name=name+'_conv2D-in')(x)
    for i in range(depth):
        for j in range(n_per_block):
            if i == depth - 1 or j < n_per_block - 1:
                strides = (1, 1)
                filters = x_dim*(i+1)
            elif i < depth - 1:
                strides = (2, 2)
                filters = x_dim*(i+2)
                
            x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same',
                    activation=activation, strides=strides,
                    name=name+'_conv2D-{}.{}'.format(i+1, j+1))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(z_dim, name=name+'_dense')(x)
    return tf.keras.models.Model(inputs=inp, outputs=x, name=name)

def BEGAN_autoencoder(size, channels=3, kernel_size=3,
                      z_dim=128, activation='elu',
                      n_per_block=2, concat=True, tanh=False, name=None):
    ''''''
    inp = x = tf.keras.layers.Input((size, size, channels), name=name+'_in')
    x = BEGAN_encoder(size, channels=channels, kernel_size=kernel_size,
                      z_dim=z_dim, x_dim=z_dim, activation=activation,
                      n_per_block=n_per_block, name=name+'_enc')(x)
    x = BEGAN_decoder(size, channels=channels, kernel_size=kernel_size,
                      z_dim=z_dim, x_dim=z_dim, activation=activation,
                      n_per_block=n_per_block,
                      concat=concat, tanh=tanh, name=name+'_dec')(x)
    return tf.keras.models.Model(inp, x)

def BEGAN_unet(size, channels=3, kernel_size=3,
               z_dim=128, activation='elu',
               n_per_block=2, tanh=False, name=None):
    ''''''
    skips = []
    inp = x = tf.keras.layers.Input((size, size, channels), name=name+'_in')
    depth = int(np.log2(size // 4))
    x = tf.keras.layers.Conv2D(z_dim, kernel_size,
                               padding='same', activation=activation,
                               name=name+'_conv2D-in')(x)
    for i in range(depth):
        skips.append(x)
        for j in range(n_per_block):
            if i == depth - 1 or j < n_per_block - 1:
                strides = (1, 1)
                filters = z_dim*(i+1)
            elif i < depth - 1:
                strides = (2, 2)
                filters = z_dim*(i+2)

            x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same',
                    activation=activation, strides=strides,
                    name=name+'_conv2D-{}.{}-enc'.format(i+1, j+1))(x)

    x = tf.keras.layers.Flatten()(x)
    x = code = tf.keras.layers.Dense(z_dim, name=name+'_dense-1')(x)
    x = tf.keras.layers.Dense(8*8*z_dim, name=name+'_dense-2')(x)
    x = tf.keras.layers.Reshape((8, 8, z_dim))(x)
    
    for i in range(depth):
        x = tf.keras.layers.concatenate([x, skips.pop(-1)], axis=-1,
                                        name=name+'_skip-'+str(i))
        for j in range(n_per_block):
            x = tf.keras.layers.Conv2D(z_dim, kernel_size,
                            activation=activation, padding='same',
                            name=name+'_conv2D-{}.{}-dec'.format(i+1, j+1))(x)
        if i < depth - 1:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                            name=name+'_upsampling-'+str(i))(x)

    x = tf.keras.layers.Conv2D(channels, kernel_size, activation=None,
                               padding='same', name=name+'_out')(x)
    out = tf.keras.layers.Activation('tanh')(x) if tanh else x
    return tf.keras.models.Model(inputs=inp, outputs=[x, code], name=name)

def DCGAN_discriminator(size, channels, kernel_size=5, hidden_dim=64,
                        activation='elu', batch_norm=True, name=None):
    ''''''
    inp = x = tf.keras.layers.Input(
        (size, size, channels), name=name+'_in')
    depth = int(np.log2(size // 4))
    for i in range(depth):
        x = tf.keras.layers.Conv2D(
            hidden_dim*(2**i), kernel_size, activation=None,
            strides=2, padding='same', name=name+'_conv2D-'+str(i))(x)
        if batch_norm and i > 0:
            x = tf.keras.layers.BatchNormalization(
                epsilon=1e-5, momentum=0.9, name=name+'_bn-'+str(i))(x)
        x = tf.keras.layers.Activation(
            activation, name=name+'_'+activation+'-'+str(i))(x)
    x = tf.keras.layers.Reshape(
        (4*4*hidden_dim*(2**i),), name=name+'_reshape')(x)
    out = tf.keras.layers.Dense(1, name=name+'_dense')(x)
    return tf.keras.models.Model(inp, out, name=name)
    

class BaseModel(tf.keras.models.Model):
    ''''''
    def stream_input(self, input_dirs, img_size, batch_size,
                     whitelist_extensions=['png', 'jpeg', 'jpg']):
        ''''''
        reader = tf.WholeFileReader()
        if type(input_dirs) == str:
            input_dirs = [input_dirs]
        x = []
        for i in input_dirs:
            imgs = []
            for ext in whitelist_extensions:
                for caps in (0, 1):
                    for descend in (0, 1):
                        pattern = '*/*.'+ext if descend else '*.'+ext
                        pattern = pattern.upper() if caps else pattern.lower()
                        pattern = os.path.join(i, pattern)
                        imgs += glob.glob(pattern)
            t = tf.train.string_input_producer(imgs)
            _, read = reader.read(t)
            decoded = tf.image.decode_png(read)
            rescaled = self.preproc_img(decoded)
            resized = tf.image.resize_images(rescaled, [img_size, img_size])
            resized = tf.cond(
                tf.equal(1, tf.shape(resized)[-1]),
                lambda: tf.tile(resized, [1,1,3]),
                lambda: resized
            )
            resized.set_shape((img_size, img_size, 3))
            x.append(resized)
        return tf.train.shuffle_batch_join(
            [x], batch_size=batch_size, 
            capacity=batch_size, 
            min_after_dequeue=0)
    
    def stage_data(self, batch, memory_gb=1, n_threads=4):
        ''''''
        with tf.device('/gpu:0'):
            dtypes = [t.dtype for t in batch]
            shapes = [t.get_shape() for t in batch]
            SA = StagingArea(dtypes, shapes=shapes, memory_limit=memory_gb*1e9)
            get, put, clear = SA.get(), SA.put(batch), SA.clear()
        tf.train.add_queue_runner(
            tf.train.QueueRunner(queue=SA, enqueue_ops=[put]*n_threads, 
                                 close_op=clear, cancel_op=clear))
        return get

    def make_summary(self, output_path, img_dict={},
                     scalar_dict={}, text_dict={}, n_images=1):
        ''''''
        summaries = []
        for k, v in img_dict.items():
            summaries.append(tf.summary.image(k, v, n_images))
        for k, v in scalar_dict.items():
            summaries.append(tf.summary.scalar(k, v))
        for k, v in text_dict.items():
            summaries.append(tf.summary.text(k, v))
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            output_path, graph=self.graph)
        return summary_op, summary_writer
    
    def save_h5(self, output_path, n):
        ''''''
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save(os.path.join(
            output_path, 'ckpt_update-{}.h5'\
            .format(str(int(n)).zfill(10))))
        
    def preproc_img(self, img):
        ''''''
        img = tf.cast(img, tf.float32)
        img -= 127.5
        img /= 127.5
        return img

    def postproc_img(self, img):
        ''''''
        img = tf.clip_by_value(img, -1, 1)
        img *= 127.5
        img += 127.5
        return tf.cast(img, tf.uint8)
    
    def decay(self, lr, step, halflife=np.inf):
        ''''''
        return lr * 0.5**tf.floor(tf.cast(step, tf.float32) / tf.cast(halflife, tf.float32))
