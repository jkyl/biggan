import tensorflow as tf
import numpy as np
import time
import tqdm
from models import *

class YGanModel(BaseModel):
    ''''''
    def __init__(self, img_size, kernel_size=3, hidden_dim=128,
                 activation='selu', batch_norm=False, tanh=False, name=None):
        ''''''
        self.img_size = img_size
        self.batch_norm = batch_norm
        with tf.variable_scope('Models'):
            self.gen = BEGAN_unet(img_size, 3, kernel_size, hidden_dim,
                                  activation, n_per_block=2, tanh=tanh,
                                  name='Gen')
            self.dec = BEGAN_decoder(img_size, 3, kernel_size, hidden_dim,
                                     hidden_dim, activation, n_per_block=2,
                                     tanh=tanh, name='Dec')
            self.disc = DCGAN_discriminator(img_size, 3, 5,
                                            hidden_dim, activation,
                                            batch_norm=batch_norm,
                                            name='Disc')
        super(BaseModel, self).__init__(
            [i for m in (self.gen, self.dec, self.disc) for i in m.inputs],
            [o for m in (self.gen, self.dec, self.disc) for o in m.outputs],
            name=name)
        self.summary()

    def G(self, x):
        return self.gen(x)[0]

    def C(self, x):
        return self.gen(x)[1]

    def AE(self, x):
        return self.dec(self.C(x))

    def D(self, x):
        return self.disc(x)
        
    def L(self, y, y_hat):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_hat*tf.ones_like(y), logits=y))

    def l2(self, y, y_hat):
        return tf.reduce_mean((y - y_hat)**2)
    
    def train(self, src_input, dest_input, output,
              lambda_z=1, lr=1e-4, lr_decay=np.inf,
              batch_size=1, epoch_size=1000):
        ''''''
        with tf.variable_scope('Input'):
            
            coord = tf.train.Coordinator()
            step = tf.Variable(0, dtype=tf.int32)
            learning = tf.keras.backend.learning_phase()
            x_src, x_dest = self.stream_input([src_input, dest_input],
                                              self.img_size, batch_size)
        
        with tf.variable_scope('Optimizer'):

            # Generated image
            x_gen = self.G(x_src)

            # Autoencoded source image
            x_ae = self.AE(x_src)

            # Recovered embedding vector
            z_r = self.C(x_src)
            z_g = self.C(x_gen)

            # Discriminator logits
            D_r = self.D(x_dest)
            D_g = self.D(x_gen)

            # Discriminator losses
            real_as_real = self.L(D_r, 1)
            fake_as_fake = self.L(D_g, 0)
            fake_as_real = self.L(D_g, 1)

            # Autoencoder losses
            L_A = self.l2(x_src, x_ae)
            L_Z = self.l2(z_r, z_g)

            # Total losses
            L_D = real_as_real + fake_as_fake
            L_G = fake_as_real + lambda_z*L_Z

            # Weights
            W_A = self.gen.trainable_weights + self.dec.trainable_weights
            W_D = self.disc.trainable_weights 
            W_G = self.gen.trainable_weights

            # Learning rate decay
            lr = self.decay(lr, step, lr_decay)

            # Optimizers
            A_opt = tf.train.AdamOptimizer(lr).minimize(L_A, var_list=W_A)
            D_opt = tf.train.AdamOptimizer(lr).minimize(L_D, var_list=W_D)
            G_opt = tf.train.AdamOptimizer(lr).minimize(L_G, var_list=W_G,
                                                        global_step=step)
            # Batch norm moving stats update
            if self.batch_norm:
                d_inputs = [tf.keras.layers.Input(tensor=i) \
                          for i in (x_dest, x_gen)]
                d_outputs = [self.disc(i) for i in d_inputs]
                super(BaseModel, self).__init__(
                    d_inputs+self.dec.inputs+self.gen.inputs,
                    d_outputs+self.dec.outputs+self.gen.outputs)
                D_opt = tf.group(D_opt, *self.updates)
            
        with tf.variable_scope('Summary'):
            imgs = dict([(i, self.postproc_img(eval(i))) for i in (
                'x_src', 'x_dest', 'x_gen', 'x_ae')])
            scalars = dict([(i, eval(i)) for i in (
                'fake_as_fake', 'real_as_real', 'fake_as_real',
                'L_A', 'L_G', 'L_D', 'L_Z', 'lr')])
        summary, writer = self.make_summary(output,
            img_dict=imgs, scalar_dict=scalars, n_images=1)
            
        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess=sess, coord=coord)
                self.save_h5(output, 0)
                self.graph.finalize()
                epoch = 1
                while not coord.should_stop():
                    print('Epoch '+str(epoch)); epoch +=1 
                    for _ in tqdm.trange(epoch_size):
                        sess.run([D_opt, A_opt], {learning: True})
                        n = sess.run([G_opt, step], {learning: True})[1]
                        if not n % 10000:
                            self.save_h5(output, n)
                        if not n % 25:
                            s = sess.run(summary, {learning: False})
                            writer.add_summary(s, n)
                            writer.flush()
        except:
            coord.request_stop()
            time.sleep(1)
            raise
                
if __name__ == '__main__':
    m = YGanModel(32, hidden_dim=64, batch_norm=False, tanh=True)
    m.train('/Users/jkyl/data/mnist_png/training', 
            '/Users/jkyl/data/img_align_celeba',
            'output/YGAN/celeba_mnist_NObatchnorm_tanh_disckernel5_hidden64',
            lambda_z=1, batch_size=4)
