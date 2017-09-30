import tensorflow as tf
import numpy as np
import time
import glob 
import os

def DilatedDenseConv2D(x, kernel_size=3, channels=128,
                       activation='elu', name=None):
    '''
    Dilated conv + activation fn + concatenation with input
    '''
    y = tf.keras.layers.Conv2D(channels, kernel_size, dilation_rate=2,
                               padding='same', activation=activation, name=name)(x)
    return tf.keras.layers.concatenate([y, x], axis=-1, name=name+'_skip')

def Pix2PixModel(img_size, kernel_size=3, hidden_dim=128,
                 activation='elu', name=None):
    '''
    Receptive field increases as 2^n, so set n = log2(size) for global
    receptive field at the output units.
    '''
    inp = x = tf.keras.layers.Input(shape=(img_size, img_size, 3)) 
    for i in range(int(np.log2(img_size))):
        x = DilatedDenseConv2D(x, kernel_size, hidden_dim,
                               activation, name=name+'_DDconv2D-'+str(i))
    out = tf.keras.layers.Conv2D(3, 1, activation=None, name=name+'_out')(x)
    return tf.keras.models.Model(inp, out, name=name)

class BaseModel(tf.keras.models.Model):
    ''''''
    def stream_input(self, input_dir, img_size, batch_size):
        ''''''
        pngs = glob.glob(os.path.join(input_dir, '*.png'))
        t_png = tf.train.string_input_producer(pngs)
        reader = tf.WholeFileReader()
        _, read = reader.read(t_png)
        decoded = tf.image.decode_png(read)
        rescaled = (tf.cast(decoded, tf.float32) - 127.5) / 127.5
        #resized = tf.image.resize_bicubic(rescaled, [img_size, img_size])
        rescaled.set_shape((img_size, img_size, 3))
        return tf.train.shuffle_batch_join(
            [[rescaled]], 
            batch_size=batch_size, 
            capacity=batch_size, 
            min_after_dequeue=0)

class CycleGanModel(BaseModel):
    ''''''
    def __init__(self, img_size, kernel_size=3, hidden_dim=128, activation='relu'):
        ''''''
        self.img_size = img_size
        self.gen_A, self.gen_B, self.disc_A, self.disc_B = models = [
            Pix2PixModel(img_size, kernel_size, hidden_dim, activation,
                         name=['G_A', 'G_B', 'D_A', 'D_B'][i])
            for i in range(4)]
        super(BaseModel, self).__init__(
            [m.input for m in models], [m.output for m in models])
        self.summary()
        
    def G_A(self, x):
        return self.gen_A(x)
    
    def G_B(self, x):
        return self.gen_B(x)
    
    def D_A(self, x):
        return tf.reduce_mean((self.disc_A(x) - x)**2)
    
    def D_B(self, x):
        return tf.reduce_mean((self.disc_B(x) - x)**2)
    
    def cycle_loss(self, A, B):
        return tf.reduce_mean((A - self.G_B(self.G_A(A)))**2)\
             + tf.reduce_mean((B - self.G_A(self.G_B(B)))**2)
     
    def grad_penalty(self, A, B):
        alpha = tf.random_uniform(
            shape=(2, tf.shape(A)[0], 1, 1, 1),
            minval=0.,
            maxval=1.
        )
        interpolates = [B + alpha[0]*(self.G_A(A) - B), A + alpha[1]*(self.G_B(B) - A)]
        gradients = [tf.gradients(self.D_B(i), [i]) for i in interpolates]
        slopes = [tf.sqrt(tf.reduce_sum(tf.square(g), axis=(1, 2, 3))) for g in gradients]
        return tf.reduce_sum([tf.reduce_mean((s-1.)**2) for s in slopes])
        
    
    def train(self, input_A, input_B,
              lambda_cyc=1, lambda_gp=1,
              batch_size=4, d_step=2):
        ''''''
        
        with tf.variable_scope('Input'):
            
            coord = tf.train.Coordinator()
            step = tf.Variable(0, dtype=tf.int32, name='global_step')
            A, B = [self.stream_input(i, self.img_size, batch_size) 
                    for i in (input_A, input_B)]
        
        with tf.variable_scope('Optimizer'):

            L_GP = self.grad_penalty(A, B)
            L_D = self.D_A(A) + self.D_B(B)\
                - self.D_A(self.G_B(B)) - self.D_B(self.G_A(A))
                
            L_D_tot = lambda_gp*L_GP + L_D

            L_C = self.cycle_loss(A, B)
            L_G = self.D_A(self.G_B(B)) + self.D_B(self.G_A(A))
            L_G_tot = lambda_cyc*L_C + L_G

            W_D = self.disc_A.trainable_weights + self.disc_B.trainable_weights
            W_G = self.gen_A.trainable_weights + self.gen_B.trainable_weights
        
            D_opt = tf.train.AdamOptimizer(1e-4,
                    beta1=0, beta2=0.9).minimize(L_D_tot, var_list=W_D)
            G_opt = tf.train.AdamOptimizer(1e-4,
                    beta1=0, beta2=0.9).minimize(L_G_tot, var_list=W_G,
                                                 global_step=step)
        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess=sess, coord=coord)
                self.save('/Users/jkyl/Desktop/cyclegan_{}.h5'.format(
                    '0'.zfill(8)))
                self.graph.finalize()
                while not coord.should_stop():
                    for _ in range(d_step):
                        _, l, g = sess.run([D_opt, L_D, L_GP])
                    _, c, n = sess.run([G_opt, L_C, step])
                    print('\nD loss: {}\nC loss: {}\nGP loss: {}'.format(l, c, g))
                    if not n % 100:
                        self.save('/Users/jkyl/Desktop/cyclegan_{}.h5'.format(
                            str(n).zfill(8)))
        except:
            coord.request_stop()
            time.sleep(1)
            raise

                
if __name__ == '__main__':
    m = CycleGanModel(32, hidden_dim=32)
    m.train('/Users/jkyl/data/cifar100/A', 
            '/Users/jkyl/data/cifar100/B',
            lambda_cyc=1, lambda_gp=10,
            batch_size=4, d_step=5)
