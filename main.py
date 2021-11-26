# For inference

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot


# load model 
model = load_model('./generator_model.h5')

#Generation of random noise to feed the generator
#z = tf.random.normal((46, 4 ),seed =10)



# loading the noise used to train our model
df = pd.read_csv('./noise.csv', header = None)
z= df.to_numpy()



# checking performance of the trained model
# prediction
gen_data = model.predict(z)
gen_data = pd.DataFrame(gen_data)
print(gen_data)

gen_data.to_csv('/content/generated_samples.csv', header=None)