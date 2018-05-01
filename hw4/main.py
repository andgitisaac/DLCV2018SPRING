import tensorflow as tf
from models.vae import VAE
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'train', 'reconstruct', 'sample' or 'encode'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_integer('batch_size', 32, "batch size")
flags.DEFINE_integer('latent_dim', 64, "dimension of latent space")
flags.DEFINE_integer('train_iter', 300000, "number of training steps")
FLAGS = flags.FLAGS

def main(_):
    
    model = VAE(mode=FLAGS.mode, batch_size=FLAGS.batch_size, latent_dim=FLAGS.latent_dim)
    solver = Solver(model, batch_size=FLAGS.batch_size, train_iter=FLAGS.train_iter, sample_iter=100, 
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
        
if __name__ == '__main__':
    tf.app.run()