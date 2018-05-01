import tensorflow as tf
import tensorflow.contrib.slim as slim

class VAE(object):
    def __init__(self, mode='train', batch_size=32, latent_dim=64, kl_weight=1e-5, learning_rate=1e-4):
        self.mode = mode
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        
    
    def encoder(self, input_, latent_dim):
        # input_: (batch, 64, 64, 3)
        with tf.variable_scope('encoder'):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                stride=2,  weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    
                    net = slim.conv2d(input_, 64, [3, 3], scope='conv1')    # (batch_size, 32, 32, 64)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2')      # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3')      # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 512, [3, 3], scope='conv4')      # (batch_size, 4, 4, 512)
                    net = slim.flatten(net, scope='flatten')

                    net = slim.fully_connected(net, latent_dim * 2, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope='fcn')
                    net = slim.batch_norm(net, activation_fn=None, scope='bn4')

                    return net
    
    def decoder(self, input_):
        # input_: (batch, ?, ?, ?)
        with tf.variable_scope('decoder'):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,
                                    stride=2,  weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    
                    dim = 8*8*256
                    net = slim.fully_connected(input_, dim, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope='fcn')
                    net = slim.batch_norm(net, scope='bn1')
                    net = tf.reshape(net, shape=[-1, 8, 8, 256])

                    net = slim.conv2d_transpose(net, 256, [3, 3], scope='deconv1')       # (batch_size, 16, 16, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d_transpose(net, 128, [3, 3], scope='deconv2')          # (batch_size, 32, 32, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d_transpose(net, 64, [3, 3], scope='deconv3')           # (batch_size, 64, 64, 64)
                    net = slim.batch_norm(net, scope='bn4')

                    net = slim.conv2d_transpose(net, 3, [3, 3], stride=1, activation_fn=tf.nn.tanh, scope='deconv4') # (batch_size, 64, 64, 3)
                    return net

    def build_model(self):
        
        if self.mode == 'train':
            with tf.variable_scope('vae'):
                self.images = tf.placeholder(tf.float32, [None, 64, 64, 3], 'input_images')
                
                with tf.variable_scope('encoder'):
                    self.encoded = self.encoder(self.images, self.latent_dim)
                
                with tf.variable_scope('latent'):
                    self.mean = self.encoded[:, :self.latent_dim]
                    self.logvar = self.encoded[:, self.latent_dim:]
                    stddev = tf.sqrt(tf.exp(self.logvar))
                    epsilon = tf.random_normal([self.batch_size, self.latent_dim])
                    self.z = self.mean + stddev * epsilon
                
                with tf.variable_scope('decoder'):
                    self.reconst = self.decoder(self.z)
                    
                    
                
                # loss
                with tf.variable_scope('loss'):
                    with tf.variable_scope('kl-divergence_loss'):
                        self.kl_loss = -0.5 * tf.reduce_sum(1 + self.logvar - tf.square(self.mean) - tf.exp(self.logvar))
                    with tf.variable_scope('reconstruction_loss'):
                        self.reconst_loss = tf.reduce_mean(tf.square(self.images - self.reconst))
                    with tf.variable_scope('total_loss'):
                        self.loss = (self.reconst_loss + self.kl_weight * self.kl_loss) / self.batch_size
               
                # optimizer
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                
                t_vars = tf.trainable_variables()
                # enc_vars = [var for var in t_vars if 'encoder' in var.name]
                # dec_vars = [var for var in t_vars if 'decoder' in var.name]
                
                # train op
                with tf.name_scope('train_op'):
                    self.train_op = slim.learning.create_train_op(self.loss, self.optimizer, variables_to_train=t_vars)
                
                # summary op
                kl_loss_summary = tf.summary.scalar('kl_loss', self.kl_loss)
                l2_loss_summary = tf.summary.scalar('l2_loss', self.reconst_loss)
                total_loss_summary = tf.summary.scalar('total_loss', self.loss)
                origin_images_summary = tf.summary.image('origin_images', self.images)
                reconstrcted_images_summary = tf.summary.image('reconstructed_images', self.reconst)
                self.summary_op = tf.summary.merge([kl_loss_summary, l2_loss_summary, 
                                                        total_loss_summary, origin_images_summary, 
                                                        reconstrcted_images_summary])                
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)

        elif self.mode == 'reconstruct' or 'sample' or 'encode':
            self.images = tf.placeholder(tf.float32, [None, 64, 64, 3], 'input_images')

            encoded = self.encoder(self.images, self.latent_dim)

            self.mean = encoded[:, :self.latent_dim]
            logvar = encoded[:, self.latent_dim:]
            stddev = tf.sqrt(tf.exp(logvar))
            epsilon = tf.random_normal([self.batch_size, self.latent_dim])
            self.z = self.mean + stddev * epsilon

            self.reconst = self.decoder(self.z)
        
    