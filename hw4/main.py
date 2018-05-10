import tensorflow as tf
from models.vae import VAE
from models.gan import GAN
from models.acgan import ACGAN
from solver.vae_solver import VAE_Solver
from solver.gan_solver import GAN_Solver
from solver.acgan_solver import ACGAN_Solver

flags = tf.app.flags
flags.DEFINE_string('network', 'vae', "'vae', 'gan', 'acgan' or 'infogan'")
flags.DEFINE_string('mode', 'train', "'train', 'reconstruct', 'sample' or 'encode'")
flags.DEFINE_string('log_save_path', 'logs', "directory for saving logs")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_integer('batch_size', 32, "batch size")
flags.DEFINE_integer('latent_dim', 64, "dimension of latent space")
flags.DEFINE_integer('train_iter', 300000, "number of training steps")
FLAGS = flags.FLAGS

def main(_):
    
    if FLAGS.network == 'vae':
        model = VAE(mode=FLAGS.mode, batch_size=FLAGS.batch_size, latent_dim=FLAGS.latent_dim)
        solver = VAE_Solver(model, batch_size=FLAGS.batch_size, train_iter=FLAGS.train_iter, log_dir=FLAGS.log_save_path,
                        model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
        
        # create directories if not exist
        if not tf.gfile.Exists(FLAGS.model_save_path):
            tf.gfile.MakeDirs(FLAGS.model_save_path)
        if not tf.gfile.Exists(FLAGS.sample_save_path):
            tf.gfile.MakeDirs(FLAGS.sample_save_path)
        
        if FLAGS.mode == 'train':
            solver.train()
        elif FLAGS.mode == 'reconstruct':
            solver.reconstruct()
        elif FLAGS.mode == 'sample':
            solver.sample()
        elif FLAGS.mode == 'encode':
            solver.encode()

    elif FLAGS.network == 'gan':
        z_dim = 100
        model = GAN(mode=FLAGS.mode)
        solver = GAN_Solver(model, batch_size=FLAGS.batch_size, z_dim=z_dim, train_iter=FLAGS.train_iter, log_dir=FLAGS.log_save_path,
                        model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
        
        # create directories if not exist
        if not tf.gfile.Exists(FLAGS.model_save_path):
            tf.gfile.MakeDirs(FLAGS.model_save_path)
        if not tf.gfile.Exists(FLAGS.sample_save_path):
            tf.gfile.MakeDirs(FLAGS.sample_save_path)
        
        if FLAGS.mode == 'train':
            solver.train()
        elif FLAGS.mode == 'sample':
            solver.sample()

    elif FLAGS.network == 'acgan':
        z_dim = 128
        feature_class = 'Smiling'
        model = ACGAN(mode=FLAGS.mode, batch_size=FLAGS.batch_size)
        solver = ACGAN_Solver(model, batch_size=FLAGS.batch_size, z_dim=z_dim, feature_class=feature_class, 
                        train_iter=FLAGS.train_iter, log_dir=FLAGS.log_save_path,
                        model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
        
        # create directories if not exist
        if not tf.gfile.Exists(FLAGS.model_save_path):
            tf.gfile.MakeDirs(FLAGS.model_save_path)
        if not tf.gfile.Exists(FLAGS.sample_save_path):
            tf.gfile.MakeDirs(FLAGS.sample_save_path)
        
        if FLAGS.mode == 'train':
            solver.train()
        elif FLAGS.mode == 'sample':
            solver.sample()
        
if __name__ == '__main__':
    tf.app.run()