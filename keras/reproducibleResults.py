import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random as rn
import os
from keras import backend as K

# this is the pre-code required for reproducible results
# TODO: single thread in TF 2.0

# set seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
rn.seed(1254)
#tf.random.set_seed(89)
tf.set_random_seed(89)

# set to single thread
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# remaining code...