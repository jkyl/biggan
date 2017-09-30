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
    #y = tf.keras.layers.BatchNormalization()(y)
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
                               activation, name=name+'_conv2D-'+str(i))
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
    def __init__(self, img_size, kernel_size=3, hidden_dim=128, activation='selu'):
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
    
    def L_D(self, A, B, norm=1):
        return tf.reduce_mean([
            tf.abs(A - self.disc_A(A))**norm,
            tf.abs(B - self.disc_B(B))**norm])

    def L_C(self, A, B, norm=1):
        return tf.reduce_mean([
            tf.abs(A - self.G_B(self.G_A(A)))**norm,
            tf.abs(B - self.G_A(self.G_B(B)))**norm])
    
    def M(self, L_D, L_G, gamma=0.5):
        return L_D + tf.abs(gamma*L_D - L_G)
    
    def update_kt(self, kt, eta, L_D, L_G, gamma=0.5):
        return tf.assign(kt, tf.clip_by_value(
            kt + eta*(gamma*L_D - L_G), 0, 1))
    
    def train(self, input_A, input_B,
              output='/home/paperspace/training/cyclegan',
              lambda_c=1, k_0=0, eta=1, gamma=.75,
              batch_size=4):
        ''''''
        with tf.variable_scope('Input'):
            
            coord = tf.train.Coordinator()
            step = tf.Variable(0, dtype=tf.int32, name='global_step')
            A, B = [self.stream_input(i, self.img_size, batch_size) 
                    for i in (input_A, input_B)]
            kt = tf.Variable(k_0, dtype=tf.float32, name='kt')
        
        with tf.variable_scope('Optimizer'):

            L_D = self.L_D(A, B)
            L_C = self.L_C(A, B)
            L_G = self.L_D(self.G_A(A), self.G_B(B))

            L_D_tot = L_D - kt*L_G
            L_G_tot = L_G + lambda_c*L_C

            W_D = self.disc_A.trainable_weights + self.disc_B.trainable_weights
            W_G = self.gen_A.trainable_weights + self.gen_B.trainable_weights
        
            D_opt = tf.train.AdamOptimizer(1e-4).minimize(L_D_tot, var_list=W_D)
            G_opt = tf.train.AdamOptimizer(1e-4).minimize(L_G_tot, var_list=W_G,
                                                          global_step=step)
            
            D_opt = tf.group(D_opt, self.update_kt(kt, eta, L_D, L_G, gamma))
            M = self.M(L_D, L_G, gamma)
            
        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess=sess, coord=coord)
                self.save(os.path.join(output, 
                    'cyclegan_{}.h5'.format('0'.zfill(8))))
                self.graph.finalize()
                while not coord.should_stop():
                    _, D, G, k, m = sess.run([D_opt, L_D, L_G, kt, M])
                    _, C, n = sess.run([G_opt, L_C, step])
                    print('\nD loss: {}\nG loss: {}\nC loss: {}\nM: {}\nk_{}: {}'\
                          .format(D, G, C, m, n, k))
                    if not n % 10000:
                        self.save(os.path.join(output, 
                            'cyclegan_{}.h5'.format(str(n).zfill(8))))
        except:
            coord.request_stop()
            time.sleep(1)
            raise
                
if __name__ == '__main__':
    m = CycleGanModel(32, hidden_dim=128)
    m.train('/home/paperspace/data/cifar100/A', 
            '/home/paperspace/data/cifar100/B',
            lambda_c=0, gamma=1, eta=0.01, batch_size=64)
