import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import load_pickle, grid_plot, grid_plot_pair, plot_tsne


class VAE_Solver(object):

    def __init__(self, model, batch_size=32, train_iter=300000,
                data_path='data', log_dir='logs',
                reconstruct_save_path='reconstruct', sample_save_path='sample',
                model_save_path='model'):
        
        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.data_path = data_path
        self.log_dir = log_dir
        self.reconstruct_save_path = reconstruct_save_path
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
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
            saver = tf.train.Saver(max_to_keep=5)

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

    def load_latest(self, saver, sess):
        if os.path.exists(os.path.join(self.model_save_path,'checkpoint')):
            print("Restoring checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))
            print("Checpoint restored!")

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
            self.load_latest(saver, sess)
            # saver.restore(sess, self.model_save_path)

            print('start reconstructing..!')
            batch_size = self.batch_size if self.batch_size <= 32 else 32
            sample_iter = images.shape[0] // batch_size


            for i in range(sample_iter):
                print("\r{}/{}".format(i+1, sample_iter), end='')
                batch_images = images[i*batch_size:(i+1)*batch_size]
                feed_dict = {model.images: batch_images}
                reconstruct = sess.run(model.reconst, feed_dict)

                output_name = os.path.join(self.reconstruct_save_path, '{:03d}.jpg'.format(i))
                print('Saving to {}'.format(output_name))
                grid_plot_pair(batch_images, reconstruct, 8, output_name)

    
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
                z = np.random.normal(size=[batch_size, model.latent_dim])
                feed_dict = {model.z: z}
                sample = sess.run(model.reconst, feed_dict)

                output_name = os.path.join(self.sample_save_path, '{:03d}.jpg'.format(i))
                print('Saving to {}'.format(output_name))
                grid_plot(sample, 8, output_name)
                
    def encode(self):
        # load dataset
        images, attrs = load_pickle(self.data_path, split='test')

        # build model
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print('loading testing model..')
            saver = tf.train.Saver()
            self.load_latest(saver, sess)

            print('start encoding..!')
            batch_size = self.batch_size if self.batch_size <= 32 else 32
            sample_iter = 30
            
            tsne_encoded = np.empty((batch_size*sample_iter, 64))
            tsne_attrs = attrs[:sample_iter*batch_size]
            for i in range(sample_iter):
                print("\r{}/{}".format(i+1, sample_iter), end='')
                batch_images = images[i*batch_size:(i+1)*batch_size]

                feed_dict = {model.images: batch_images}
                batch_encoded = sess.run(model.mean, feed_dict)
                tsne_encoded[i*batch_size:(i+1)*batch_size, :] = batch_encoded
            plot_tsne(tsne_encoded, tsne_attrs)

                
