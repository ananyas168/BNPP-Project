# -*- coding: utf-8 -*-
"""GAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_bL49CQOuEiqXUpMoV0HrK51fd7nyCii
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/BNPP_Project/tabgan/


# loading the training data
path = '/content/drive/MyDrive/BNPP_Project/outcoder_gan_submission2/'

import pandas as pd
df0 = pd.read_csv(path+'train.csv',header= None)
df1 =  pd.read_csv(path+'data_val_log_return.csv',header= None)
df= pd.concat([df0,df1],axis = 0).reset_index(drop=True)
print(df)
print(df.shape)
df = df.drop(columns=[0],axis = 1)
print(df)
print(df.shape)

#df = df.iloc[:700,:]
df

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from keras.regularizers import l1
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam
path_model= path +'model/'
path_noise=path +'noise/'
path_gen = path +'gen_data/'
path_weight= path +'weights/'
if not os.path.exists(path_model):
    os.makedirs(path_model)

if not os.path.exists(path_model):
    os.makedirs(path_model)

if not os.path.exists(path_noise):
    os.makedirs(path_noise)

if not os.path.exists(path_gen):
    os.makedirs(path_gen)
if not os.path.exists(path_weight):
    os.makedirs(path_weight)

class GAN():
    
    def __init__(self, gan_args):
        [self.batch_size, lr, self.noise_dim,
         self.data_dim, layers_dim] = gan_args

        self.generator = Generator(self.batch_size).\
            build_model(input_shape=(self.noise_dim,), dim=layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(self.batch_size).\
            build_model(input_shape=(self.data_dim,), dim=layers_dim)

        optimizer = Adam(lr, 0.5)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        record = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(record)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_data_batch(self, train, batch_size, seed=0):
        # # random sampling - some samples will have excessively low or high sampling, but easy to implement
        # np.random.seed(seed)
        # x = train.loc[ np.random.choice(train.index, batch_size) ].values
        # iterate through shuffled indices, so every sample gets covered evenly

        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        train_ix = np.random.choice(list(train.index), replace=False, size=len(train))  # wasteful to shuffle every time
        train_ix = list(train_ix) + list(train_ix)  # duplicate to cover ranges past the end of the set
        x = train.loc[train_ix[start_i: stop_i]].values
        #print(len(train_ix)
        return np.reshape(x, (batch_size, -1))
        
    def train(self, data, train_arguments):
        [cache_prefix, epochs, sample_interval] = train_arguments
        
        data_cols = data.columns

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            batch_data = self.get_data_batch(data, self.batch_size)
            noise = tf.random.normal((self.batch_size, self.noise_dim), seed = 123)
            # Generate a batch of new data points
            gen_data = self.generator.predict(noise)
    
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(batch_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
            # ---------------------
            #  Train Generator
            # ---------------------
            noise = tf.random.normal((self.batch_size, self.noise_dim), seed =123)
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
    
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
    
            # If at save interval => save generated events
            if epoch % sample_interval == 0 or epoch == epochs-1:
                #Test here data generation step
                # save model checkpoints path_weight
                # model_checkpoint_base_name = 'model/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                # model_checkpoint_base_name1 = 'model/model/' + cache_prefix + '_{}_model_step_{}.h5'
                model_checkpoint_base_name = 'weights/' + cache_prefix + '_{}_model_weights_step_{}.h5'
                model_checkpoint_base_name1 = 'model/' + cache_prefix + '_{}_model_step_{}.h5'
                self.generator.save(model_checkpoint_base_name1.format('generator', epoch))
                self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))
                if epoch == epochs-1:
                  model_checkpoint_base_name11 = 'model/' + cache_prefix + '_{}_model_final.h5'
                  self.generator.save(model_checkpoint_base_name11.format('generator'))

                #Here is generating the data
                z = tf.random.normal((410, self.noise_dim ),seed =123)
                gen_data = self.generator(z)
                print('gen_data',gen_data.shape)
                print('generated_data')
                # if epoch == epochs-1:
                #   np.savetxt(path_noise+'noise_final.csv',z,delimiter=',')
                #   np.savetxt(path_gen+'/gen_final.csv',gen_data, delimiter=',')
                #   model_checkpoint_base_name11 = 'model/' + cache_prefix + '_{}_model_final.h5'
                #   self.generator.save(model_checkpoint_base_name11.format('generator'))
                #   print(gen_data)
                #   print('noise':z)
                # else:
                np.savetxt(path_noise+'/noise_{}.csv'.format(epoch),z,delimiter=',')
                np.savetxt(path_gen+'/saved{}.csv'.format(epoch),gen_data, delimiter=',')  
        print('gen_data',gen_data.shape)
        np.savetxt(path_gen+'gen_model_final.csv',gen_data,delimiter=',')
        # model_checkpoint_base_name11 = 'model/model/' + cache_prefix + '_{}_model_final.h5'
        # self.generator.save(model_checkpoint_base_name1.format('generator'))
        #np.savetxt('/content/drive/MyDrive/BNPP_Project/tabgan/model/model/noise_final.csv',z,delimiter=',')
    def save(self, path, name):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return
    
    def load(self, path):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(path)
        return self.generator
    
class Generator():
    def __init__(self, batch_size):
        self.batch_size=batch_size
        
    def build_model(self, input_shape, dim, data_dim):
        input= Input(shape=input_shape, batch_size=self.batch_size)
       
        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(dim*2 , activation='relu')(x)
        #x = Dense(dim*2 , activation='relu')(x)# added extra after 3 rd submission
        x = Dense(data_dim,)(x) # add l1 norm
        x = tf.math.abs(x)
        return Model(inputs=input, outputs=x)

class Discriminator():
    def __init__(self,batch_size):
        self.batch_size=batch_size
    
    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 6, activation='relu')(input)
        x = Dense(dim * 4, activation='tanh')(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=input, outputs=x)

print(df.shape[1])

noise_dim = 32
dim = 32
batch_size = 32

log_step = 100
epochs = 10
learning_rate = 5e-4
'''
where:
gan_args = [batch_size, learning_rate, noise_dim, df.shape[1], dim]
train_args = ['', epochs, log_step]
''' 

gan_args = [1024, 3e-4, 4, df.shape[1], 32]# batchsize was 64 now 128
train_args = ['',24 , 100]

model = GAN

#Training the GAN model chosen: Vanilla GAN, CGAN, DCGAN, etc.
synthesizer = model(gan_args)
synthesizer.train(df, train_args)
# gan_args[1]=9e-5
# train_args[1]=80
# synthesizer.train(df, train_args)
gan_args[1]=6e-7
train_args[1]=500
synthesizer.train(df, train_args)
synthesizer.generator.summary()

synthesizer.discriminator.summary()