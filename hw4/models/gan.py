import tensorflow as tf
import tensorflow.contrib.slim as slim

class GAN(object):
    def __init__(self, mode='train', z_dim=100, learning_rate=1e-4):
        self.mode = mode
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def batch_norm_params(self, is_training):
        return {
            "decay": 0.9,
            "epsilon": 1e-5,
            "scale": True,
            "updates_collections": None,
            "is_training": is_training
        }

    def generator(self, input_, reuse=False):
        # input_: (batch, self.z_dim)
        
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params(self.mode=='train'),
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                with slim.arg_scope([slim.conv2d_transpose],
                                    kernel_size=[5, 5],
                                    stride=2, padding="SAME"):

                    net = slim.fully_connected(input_, 4*4*512, normalizer_fn=None, normalizer_params=None, scope="projection")
                    net = tf.reshape(net, [-1, 4, 4, 512])  
                    net = slim.batch_norm(net, **self.batch_norm_params(self.mode=='train'), scope="batch_norm")     

                    net = slim.conv2d_transpose(net, 256, scope='conv_transpose1')          # (batch_size, 8, 8, 256)
                    net = slim.conv2d_transpose(net, 128, scope='conv_transpose2')          # (batch_size, 16, 16, 128)
                    net = slim.conv2d_transpose(net, 64, scope='conv_transpose3')           # (batch_size, 32, 32, 64)
                    net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh,
                                                    normalizer_fn=None,
                                                    normalizer_params=None,
                                                    scope='conv_transpose4')                # (batch_size, 64, 64, 3)
                    return net

    def discriminator(self, input_, reuse=False):
        # input_: (batch, 64, 64, 3)

        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params(self.mode=='train'),
                                kernel_size=[5, 5],
                                stride=2, padding="SAME"):
                
                net = slim.conv2d(input_, 64,
                                    normalizer_fn=None,
                                    normalizer_params=None,
                                    scope="conv1")                  # (batch_size, 32, 32, 64)                    
                net = slim.conv2d(net, 128, scope='conv2')          # (batch_size, 16, 16, 128)
                net = slim.conv2d(net, 256, scope='conv3')          # (batch_size, 8, 8, 256)
                net = slim.conv2d(net, 512, scope='conv4')          # (batch_size, 4, 4, 512)
                net = slim.conv2d(net, 1, 
                                    activation_fn=None,
                                    kernel_size=[4, 4], 
                                    stride=1, padding='VALID',
                                    normalizer_fn=None,
                                    normalizer_params=None,
                                    scope='conv5')                  # (batch_size, 1, 1, 1)
                net = slim.flatten(net)
                # net = tf.squeeze(net, [1, 2], name="squeeze")
                # print(net.get_shape().as_list())
                return net 

    def build_model(self):
        
        if self.mode == 'train':
            with tf.variable_scope('gan'):
                self.real_images = tf.placeholder(tf.float32, [None, 64, 64, 3], 'real_images')
                self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='input_z')                
                
                self.fake_images = self.generator(self.z)

                self.d_real = self.discriminator(self.real_images)
                self.d_fake = self.discriminator(self.fake_images, reuse=True)                

                # loss
                with tf.variable_scope('Loss_D'):
                    with tf.variable_scope('Loss_D_real'):
                        self.loss_d_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.d_real), logits=self.d_real)
                    with tf.variable_scope('Loss_D_fake'):
                        self.loss_d_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.d_fake), logits=self.d_fake)
                    with tf.variable_scope('Loss_D_total'):
                        self.loss_d = self.loss_d_real + self.loss_d_fake
                
                with tf.variable_scope('Loss_G'):
                    self.loss_g = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.d_fake), logits=self.d_fake)

                # accuracy
                # reference: https://github.com/gitlimlab/SSGAN-Tensorflow/blob/master/model.py
                with tf.variable_scope('Accuracy_D'):
                    with tf.variable_scope('Accuracy_D_real'):
                        self.acc_d_real = tf.reduce_mean(tf.cast(self.d_real > .5, tf.float32))
                    with tf.variable_scope('Accuracy_D_fake'):
                        self.acc_d_fake = tf.reduce_mean(tf.cast(self.d_real < .5, tf.float32))

                # optimizer
                t_vars = tf.trainable_variables()

                with tf.variable_scope("Optimizer_D"):
                    d_vars = [var for var in t_vars if 'discriminator' in var.name]
                    self.d_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_d, global_step=self.global_step, var_list=d_vars)

                with tf.variable_scope("Optimizer_G"):
                    g_vars = [var for var in t_vars if 'generator' in var.name]
                    self.g_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_g, var_list=g_vars)
               
                # summary op
                loss_d_summary = tf.summary.scalar('loss_d', self.loss_d)
                loss_d_real_summary = tf.summary.scalar('loss_d_real', self.loss_d_real)
                loss_d_fake_summary = tf.summary.scalar('loss_d_fake', self.loss_d_fake)
                loss_g_summary = tf.summary.scalar('loss_g', self.loss_g)
                acc_d_real_summary = tf.summary.scalar('acc_d_real', self.acc_d_real)
                acc_d_fake_summary = tf.summary.scalar('acc_d_fake', self.acc_d_fake)
                generated_images_summary = tf.summary.image('generated_images', self.fake_images, max_outputs=9)
                
                self.summary_op = tf.summary.merge([loss_d_summary, 
                                                        loss_d_real_summary, 
                                                        loss_d_fake_summary,
                                                        loss_g_summary,
                                                        acc_d_real_summary,
                                                        acc_d_fake_summary,
                                                        generated_images_summary])
                
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)

        elif self.mode == 'sample':
            with tf.variable_scope('gan'):
                self.z = tf.placeholder(tf.float32, [None, self.z_dim], name="input_z")
                
                with tf.variable_scope('generator'):
                    self.fake_images = self.generator(self.z)


        