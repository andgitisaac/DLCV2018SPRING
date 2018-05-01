import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import load_pickle


class Solver(object):

    def __init__(self, model, batch_size=32, train_iter=300000, sample_iter=100,
                data_path='data', log_dir='logs', sample_save_path='sample',
                model_save_path='model', test_model='model/vae-300000'):
        
        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.data_path = data_path
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True

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
                    print("Restoring checkpoint")
                    saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))
            load_latest()

            print ('Start training..!')
            for step in range(self.train_iter+1):
                start = time.time()
                i = step % int(images.shape[0] // self.batch_size)
                
                batch_images = images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}

                sess.run(model.train_op, feed_dict)      
                
                if (step+1) % 50 == 0:
                    summary, kl, l2, loss = sess.run([model.summary_op,
                                                    model.kl_loss,
                                                    model.reconst_loss,
                                                    model.loss],
                                                    feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] kl_loss: [%.5f] l2_loss: [%.5f] total_loss: [%.5f] Time: [%.5f]' \
                               %(step+1, self.train_iter, kl, l2, loss, time.time() - start))
                
                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'vae'), global_step=step+1)
                    print ('model/vae-%d saved' %(step+1))
                
    def reconstruct(self):
        # load dataset
        images, _ = load_pickle(self.data_path, split='test')

        # build model
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print('loading testing model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print('start reconstructing..!')
            for i in range(self.sample_iter):
                batch_images = images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                reconstruct = sess.run(model.reconst, feed_dict)
    
    def sample(self):
        # build model
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print('loading testing model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print('start sampling..!')
            for _ in range(self.sample_iter):
                z = np.random.normal(size=[self.batch_size, model.latent_dim])
                feed_dict = {model.z: z}
                sample = sess.run(model.reconst, feed_dict)

    def encode(self):
        # load dataset
        images, _ = load_pickle(self.data_path, split='test')

        # build model
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print('loading testing model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print('start encoding..!')
            for i in range(self.sample_iter):
                batch_images = images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                encoded = sess.run(model.mean, feed_dict)
