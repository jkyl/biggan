import tensorflow as tf
import argparse

from src import nets, data

def model_fn(features, labels, mode, params):
  del labels # unused

  # set the learning phase and float precision
  tf.keras.backend.set_learning_phase(True)
  tf.keras.backend.set_floatx(params['dtype'])
  features = tf.cast(features, params['dtype'])

  # build the networks
  G = nets.Generator(params['channels'])
  D = nets.Discriminator(params['channels'])
  G.summary(); D.summary()

  # sample z from N(0, 1)
  z = tf.random.normal((
    tf.shape(features)[0], 128), 
      dtype=params['dtype'])

  # make predictions
  predictions = G(z)
  logits_real = D(features)
  logits_fake = D(predictions)

  # hinge loss function
  L_G = -tf.reduce_mean(logits_fake)
  L_D = tf.reduce_mean(tf.nn.relu(1. - logits_real))\
      + tf.reduce_mean(tf.nn.relu(1. + logits_fake))

  # two-timescale update rule
  G_adam = tf.optimizers.Adam(1e-4, 0., 0.999, 1e-4)
  D_adam = tf.optimizers.Adam(4e-4, 0., 0.999, 1e-4)
  
  # gradients
  grad_G = G_adam.get_gradients(L_G, G.trainable_weights)
  grad_D = D_adam.get_gradients(L_D, D.trainable_weights)

  # nD = 1
  train_op = tf.group(
    G_adam.apply_gradients(zip(grad_G, G.trainable_weights)),
    D_adam.apply_gradients(zip(grad_D, D.trainable_weights)),
    tf.compat.v1.train.get_global_step().assign(G_adam.iterations))

  # create some tensorboard summaries
  tf.compat.v1.summary.image('xhat', data.postprocess_img(predictions), 5)
  tf.compat.v1.summary.image('x', data.postprocess_img(features), 5)
  tf.compat.v1.summary.scalar('L_G', L_G)
  tf.compat.v1.summary.scalar('L_D', L_D)
 
  # return an EstimatorSpec
  return tf.estimator.EstimatorSpec(
    mode=mode, loss=L_D, train_op=train_op)

def main(args):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  tf.estimator.Estimator(
    model_fn=model_fn,
    params=vars(args),
    config=tf.estimator.RunConfig(
      train_distribute=data.get_strategy(),
      model_dir=args.model_dir)
  ).train(data.get_train_data, steps=1000000)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('data_file', type=str,
    help='.npz file containing preprocessed image data')
  p.add_argument('model_dir', type=str,
    help='directory in which to save checkpoints and summaries')
  p.add_argument('-bs', '--batch_size', type=int, default=64,
    help='number of samples per minibatch update')
  p.add_argument('-ch', '--channels', type=int, default=16,
    help='channel multiplier in G and D')
  p.add_argument('-dt', '--dtype', choices=('float32', 'float16'),
    default='float16', help='training float precision')
  main(p.parse_args())
