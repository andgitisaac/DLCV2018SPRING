import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import load_pickle, grid_plot


class GAN_Solver(object):

    def __init__(self, model, batch_size=32, 
                z_dim=100, train_iter=300000,
                data_path='data', log_dir='logs',
                sample_save_path='sample',
                model_save_path='model'):
        
        self.model = model
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.train_iter = train_iter
        self.data_path = data_path
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
    
    def generate_z(self, batch_size, z_dim):
        return np.random.uniform(-1, 1, size=(batch_size, z_dim)).astype(np.float32)

    def train(self):
        # load dataset
        images, _ = load_pickle(self.data_path, split='train')

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        with tf.Session(config=self.config) as sess:
            # initialize
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            saver = tf.train.Saver(max_to_keep=20)

            def load_latest():
                if os.path.exists(os.path.join(self.model_save_path,'checkpoint')):
                    print("Restoring checkpoint from {}".format(self.model_save_path))
                    saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))
            load_latest()

            print ('Start training..!')
            for step in range(self.train_iter+1):
                start = time.time()
                i = step % int(images.shape[0] // self.batch_size)
                
                if (step+1) % 1000 == 0:
                    # Maybe need to shuffle images?
                    np.random.shuffle(images)
                batch_images = images[i*self.batch_size:(i+1)*self.batch_size]
                
                z = self.generate_z(self.batch_size, self.z_dim)
                feed_dict = {model.z: z, model.real_images:batch_images}

                # train D
                sess.run(model.d_train_op, feed_dict=feed_dict)

                # train G
                sess.run(model.g_train_op, feed_dict=feed_dict)
                sess.run(model.g_train_op, feed_dict=feed_dict)
                sess.run(model.g_train_op, feed_dict=feed_dict)

                # print("D_REAL: ", sess.run(model.d_real, feed_dict))
                # print("D_FAKE: ", sess.run(model.d_fake, feed_dict))
                if (step+1) % 20 == 0:
                    summary, loss_d, loss_g, acc_real, acc_fake = sess.run([model.summary_op,
                                                                        model.loss_d,
                                                                        model.loss_g,
                                                                        model.acc_d_real,
                                                                        model.acc_d_fake],
                                                                        feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss_d: [%.5f] loss_g: [%.5f] acc_real: [%.5f] acc_fake: [%.5f] Time: [%.5f]' \
                               %(step+1, self.train_iter, loss_d, loss_g, acc_real, acc_fake, time.time() - start))
                
                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'gan'), global_step=step+1)
                    print ('model/gan-%d saved' %(step+1))

    def load_latest(self, saver, sess):
        if os.path.exists(os.path.join(self.model_save_path,'checkpoint')):
            print("Restoring checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))
            print("Checpoint restored!")

    
    def sample(self):
        # build model
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print('loading testing model..')
            saver = tf.train.Saver()
            self.load_latest(saver, sess)

            batch_size = self.batch_size if self.batch_size <= 32 else 32
            print('start sampling..!')
            for i in range(64):
                z = self.generate_z(self.batch_size, self.z_dim)
                feed_dict = {model.z: z}

                sample = sess.run(model.fake_images, feed_dict)

                output_name = os.path.join(self.sample_save_path, '{:03d}.jpg'.format(i))
                print('Saving to {}'.format(output_name))
                grid_plot(sample, 8, output_name)

                
