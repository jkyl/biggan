import tensorflow as tf
import numpy as np
import time
import glob 
import os
from models import *

class CycleGanModel(BaseModel):
    ''''''
    def __init__(self, img_size, kernel_size=3, hidden_dim=128, activation='selu'):
        ''''''
        self.img_size = img_size
        
        with tf.variable_scope('Generators'):
            self.G_A, self.G_B = generators = [
            BEGAN_unet(img_size, 3, kernel_size, hidden_dim, activation,
                       n_per_block=2, name=['G_A', 'G_B'][i])
            #DilatedDenseNet(img_size, 3, kernel_size, hidden_dim, activation,
            #                name=['G_A', 'G_B'][i])
            for i in range(2)]
            self.G_A.summary()
            
        with tf.variable_scope('Discriminators'):
            self.D_A, self.D_B = discriminators = [
            BEGAN_autoencoder(img_size, 3, kernel_size, hidden_dim, activation, 
                              n_per_block=2, concat=True,
                              name=['D_A', 'D_B'][i])
            for i in range(2)]
            
        super(BaseModel, self).__init__(
            [m.input for m in generators + discriminators],
            [m.output for m in generators + discriminators])
        self.summary()
        
    def L(self, x, y, norm=1):
        if norm not in (1, 2):
            raise NotImplementedError
        return tf.reduce_mean(tf.abs(x - y)**norm)

    def M(self, L_D, L_G, gamma):
        return L_D + tf.abs(gamma*L_D - L_G)
    
    def update_kt(self, kt, eta, L_D, L_G, gamma):
        return tf.assign(kt, tf.clip_by_value(
            kt + eta*(gamma*L_D - L_G), 0, 1))
    
    def train(self, input_A, input_B, output,
              lambda_c=1, k_0=0, eta=0.01, gamma=0.5,
              norm=1, batch_size=1):
        ''''''
        with tf.variable_scope('Input'):
            
            coord = tf.train.Coordinator()
            step = tf.Variable(0, dtype=tf.int32)
            kt_A, kt_B = [tf.Variable(k_0, dtype=tf.float32) for _ in (0, 1)]
            A, B = self.stream_input([input_A, input_B],
                                     self.img_size, batch_size) 
        
        with tf.variable_scope('Optimizer'):

            # B onto A and A onto B
            G_A_B = self.G_B(B)
            G_B_A = self.G_A(A)

            # Real discriminator outputs
            D_A_R = self.D_A(A)
            D_B_R = self.D_B(B)

            # Generated discriminator outputs
            D_A_G = self.D_A(G_A_B)
            D_B_G = self.D_B(G_B_A)
            
            # Cycle outputs
            G_A_B_A = self.G_B(G_B_A)
            G_B_A_B = self.G_A(G_A_B)

            # Real discriminator losses
            L_D_A = self.L(A, D_A_R, norm=norm)
            L_D_B = self.L(B, D_B_R, norm=norm)

            # Cycle losses
            L_C_A = self.L(A, G_A_B_A, norm=norm)
            L_C_B = self.L(B, G_B_A_B, norm=norm)

            # Generator losses
            L_G_A = self.L(G_A_B, D_A_G, norm=norm)
            L_G_B = self.L(G_B_A, D_B_G, norm=norm)

            # Total losses
            L_D_tot = L_D_A + L_D_B - kt_A*L_G_A - kt_B*L_G_B
            L_G_tot = L_G_A + L_G_B + lambda_c*(L_C_A + L_C_B)
            M = self.M(L_D_tot, L_G_tot, gamma)

            # Weights
            W_D = self.D_A.trainable_weights + self.D_B.trainable_weights
            W_G = self.G_A.trainable_weights + self.G_B.trainable_weights

            # Optimizers
            D_opt = tf.train.AdamOptimizer(1e-4).minimize(L_D_tot, var_list=W_D)
            G_opt = tf.train.AdamOptimizer(1e-4).minimize(L_G_tot, var_list=W_G,
                                                          global_step=step)
            # Feedback updates
            kt_A_update = self.update_kt(kt_A, eta, L_D_A, L_G_A, gamma)
            kt_B_update = self.update_kt(kt_B, eta, L_D_B, L_G_B, gamma)

            # Group them with D optimizer
            D_opt = tf.group(D_opt, kt_A_update, kt_B_update)
            
        with tf.variable_scope('Summary'):
            imgs = dict([(i, self.postproc_img(eval(i))) for i in (
                'A', 'B',
                'G_A_B', 'G_B_A',
                'D_A_R', 'D_B_R',
                'D_A_G', 'D_B_G',
                'G_A_B_A', 'G_B_A_B')])
            scalars = dict([(i, eval(i)) for i in (
                'M',
                'L_D_A', 'L_D_B',
                'L_G_A', 'L_G_B',
                'L_C_A', 'L_C_B',
                'L_D_tot', 'L_G_tot',
                'kt_A', 'kt_B')])
        summary, writer = self.make_summary(output,
            img_dict=imgs, scalar_dict=scalars, n_images=1)
            
        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess=sess, coord=coord)
                self.save_h5(output, 0)
                self.graph.finalize()
                while not coord.should_stop():
                    sess.run([D_opt])
                    n = sess.run([G_opt, step])[1]
                    if not n % 10000:
                        self.save_h5(output, n)
                    if not n % 25:
                        s = sess.run(summary)
                        writer.add_summary(s, n)
                        writer.flush()
        except:
            coord.request_stop()
            time.sleep(1)
            raise
                
if __name__ == '__main__':
    m = CycleGanModel(32, hidden_dim=32)
    m.train('/Users/jkyl/data/mnist_png/training', 
            '/Users/jkyl/data/cifar100/train',
            'output/UNET_gamma0.5_cycle0.1_bs1',
            lambda_c=0.1, gamma=0.5, eta=0.01, k_0=0, norm=1, batch_size=1)
