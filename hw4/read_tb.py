from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tags', type=str,
                        dest='tags', help='scalar tags', 
                        required=True)
parser.add_argument('--model', type=str,
                        dest='model', help='model type', 
                        required=True)
args = parser.parse_args()


tags = args.tags
model = args.model

ea = EventAccumulator('{}_logs/'.format(model))
ea.Reload()
# print(ea.Tags())
# print(ea.Scalars('gan/acc_d_fake'))
pd.DataFrame(ea.Scalars('{}/{}'.format(model, tags))).to_csv('{}.csv'.format(tags))