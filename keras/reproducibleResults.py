import numpy as np
import tensorflow as tf
import random as rn
import os
from keras import backend as K

# this is the pre-code required for reproducible results

# set seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(89)

# set to single thread
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# remaining code...