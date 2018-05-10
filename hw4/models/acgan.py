import tensorflow as tf
import tensorflow.contrib.slim as slim

class ACGAN(object):
    def __init__(self, mode='train', batch_size=32, num_classes=2, z_dim=128, learning_rate=1e-4):
        self.mode = mode
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.class_loss_weight = 1

    def batch_norm_params(self, is_training):
        return {
            "decay": 0.9,
            "epsilon": 1e-5,
            "scale": True,
            "updates_collections": None,
            "is_training": is_training
        }

    def generator(self, input_, reuse=False):
        # input_: (batch, [self.num_classes + self.z_dim])
        
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

    def discriminator(self, input_, num_classes, reuse=False):
        # input_: (batch, 64, 64, 3)

        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params(self.mode=='train'),
                                kernel_size=[5, 5],
                                stride=2, padding="SAME"):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.leaky_relu,
                                    normalizer_fn=None,
                                    normalizer_params=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):                
                    net = slim.conv2d(input_, 64,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="conv1")                  # (batch_size, 32, 32, 64)                    
                    net = slim.conv2d(net, 128, scope='conv2')          # (batch_size, 16, 16, 128)
                    recog_class = slim.flatten(net)
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

                    recog_class = slim.fully_connected(recog_class, 1024, scope="predict_class_fc1")
                    recog_class = slim.fully_connected(recog_class, 128, scope="predict_class_fc2")
                    recog_class = slim.fully_connected(recog_class, num_classes, scope="predict_class_output")
                    # net = tf.squeeze(net, [1, 2], name="squeeze")
                    # print(net.get_shape().as_list())
                    return net, recog_class



    def build_model(self):
        
        if self.mode == 'train':
            with tf.variable_scope('acgan'):
                self.real_images = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_images')
                self.real_labels = tf.placeholder(tf.float32, [None, self.num_classes], name='real_label')
                self.flip_labels = tf.placeholder(tf.bool, name='label_is_flip')
                
                # concatenate noise and one-hot labels together
                self.noise = tf.placeholder(tf.float32, [None, self.z_dim], name='sample_noise')
                self.sample_labels =  tf.placeholder(tf.float32, [None, self.num_classes], name='sample_label')
                self.z = tf.concat(axis=1, values=[self.sample_labels, self.noise])            
                
                self.fake_images = self.generator(self.z)

                self.d_real, self.cls_real = self.discriminator(self.real_images, self.num_classes)
                self.d_fake, self.cls_fake = self.discriminator(self.fake_images, self.num_classes, reuse=True) 
                        

                # loss
                with tf.variable_scope('Loss_Aux'):
                    self.loss_cls_real = tf.losses.softmax_cross_entropy(logits=self.cls_real, onehot_labels=self.real_labels)
                    self.loss_cls_fake = tf.losses.softmax_cross_entropy(logits=self.cls_fake, onehot_labels=self.sample_labels)
                    # self.loss_cls_real = tf.losses.sparse_softmax_cross_entropy(logits=self.cls_real, labels=self.real_labels)
                    # self.loss_cls_fake = tf.losses.sparse_softmax_cross_entropy(logits=self.cls_fake, labels=self.sample_labels)
                    self.loss_cls = (self.loss_cls_real + self.loss_cls_fake) / 2.

                with tf.variable_scope('Loss_D'):
                    real_labels = tf.cond(self.flip_labels, true_fn=(lambda: tf.zeros_like(self.d_real)), false_fn=(lambda: tf.ones_like(self.d_real)))
                    fake_labels = tf.cond(self.flip_labels, true_fn=(lambda: tf.ones_like(self.d_fake)), false_fn=(lambda: tf.zeros_like(self.d_fake)))


                    with tf.variable_scope('Loss_D_real'):
                        # self.loss_d_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.d_real), logits=self.d_real)
                        self.loss_d_real = tf.losses.sigmoid_cross_entropy(multi_class_labels=real_labels, logits=self.d_real)
                    with tf.variable_scope('Loss_D_fake'):
                        # self.loss_d_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.d_fake), logits=self.d_fake)                        self.loss_d_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.d_fake), logits=self.d_fake)                        self.loss_d_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.d_fake), logits=self.d_fake)
                        self.loss_d_fake = tf.losses.sigmoid_cross_entropy(multi_class_labels=fake_labels, logits=self.d_fake)
                    with tf.variable_scope('Loss_D_total'):
                        self.loss_d = self.loss_d_real + self.loss_d_fake + self.class_loss_weight * self.loss_cls
                
                with tf.variable_scope('Loss_G'):
                    self.loss_g_gen = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.d_fake), logits=self.d_fake)
                    self.loss_g = self.loss_g_gen + self.class_loss_weight * self.loss_cls

                # self.soft_cls_fake = tf.nn.softmax(self.cls_fake)
                # self.soft_cls_sample = self.sample_labels

                # accuracy
                # reference: https://github.com/gitlimlab/SSGAN-Tensorflow/blob/master/model.py
                with tf.variable_scope('Accuracy_D'):
                    with tf.variable_scope('Accuracy_D_real'):
                        sigmoid_d_real = tf.nn.sigmoid(self.d_real)
                        self.acc_d_real = tf.reduce_mean(tf.cast(sigmoid_d_real > .5, tf.float32))
                    with tf.variable_scope('Accuracy_D_fake'):
                        sigmoid_d_fake = tf.nn.sigmoid(self.d_fake)
                        self.acc_d_fake = tf.reduce_mean(tf.cast(sigmoid_d_fake < .5, tf.float32))
                
                with tf.variable_scope('Accuracy_Aux'):
                    with tf.variable_scope('Accuracy_Aux_real'):
                        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.cls_real), 1), tf.argmax(self.real_labels, 1))
                        self.acc_cls_real = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    with tf.variable_scope('Accuracy_Aux_fake'):
                        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.cls_fake), 1), tf.argmax(self.sample_labels, 1))
                        self.acc_cls_fake = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
                # loss_d_real_summary = tf.summary.scalar('loss_d_real', self.loss_d_real)
                # loss_d_fake_summary = tf.summary.scalar('loss_d_fake', self.loss_d_fake)
                loss_g_summary = tf.summary.scalar('loss_g', self.loss_g)
                loss_aux_summary = tf.summary.scalar('loss_aux', self.loss_cls)
                acc_d_real_summary = tf.summary.scalar('acc_d_real', self.acc_d_real)
                acc_d_fake_summary = tf.summary.scalar('acc_d_fake', self.acc_d_fake)
                acc_aux_real_summary = tf.summary.scalar('acc_aux_real', self.acc_cls_real)
                acc_aux_fake_summary = tf.summary.scalar('acc_aux_fake', self.acc_cls_fake)               
                generated_images_summary = tf.summary.image('generated_images', self.fake_images, max_outputs=9)
                
                self.summary_op = tf.summary.merge([loss_d_summary, 
                                                        # loss_d_real_summary, 
                                                        # loss_d_fake_summary,
                                                        loss_g_summary,
                                                        loss_aux_summary,
                                                        acc_d_real_summary,
                                                        acc_d_fake_summary,
                                                        acc_aux_real_summary,
                                                        acc_aux_fake_summary,
                                                        generated_images_summary])
                
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)

        elif self.mode == 'sample':
            with tf.variable_scope('acgan'):                
                # concatenate noise and one-hot labels together
                self.noise = tf.placeholder(tf.float32, [None, self.z_dim], name='sample_noise')
                self.sample_labels =  tf.placeholder(tf.float32, [None, 2], name='sample_label')
                self.z = tf.concat(axis=1, values=[self.sample_labels, self.noise])            
                
                self.fake_images = self.generator(self.z)


        