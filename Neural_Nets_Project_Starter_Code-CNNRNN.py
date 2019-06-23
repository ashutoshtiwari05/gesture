
# coding: utf-8

# # Gesture Recognition
# In this group project, you are going to build a 3D Conv model that will be able to predict the 5 gestures correctly. Please import the following libraries to get started.

# In[1]:


import numpy as np
import os
from scipy.misc import imread, imresize
import datetime
import cv2


# We set the random seed so that the results don't vary drastically.

# In[2]:


np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.set_random_seed(30)


# In this block, you read the folder names for training and validation. You also set the `batch_size` here. Note that you set the batch size in such a way that you are able to use the GPU in full capacity. You keep increasing the batch size until the machine throws an error.

# In[3]:


train_doc = np.random.permutation(open('Project_data/train.csv').readlines())
val_doc = np.random.permutation(open('Project_data/val.csv').readlines())
batch_size = 7   #experiment with the batch size


# In[4]:


rows = 120   # X 
cols = 120   # Y 
channel = 3  # number of channels in images 3 for color(RGB)
frames=15


# In[5]:


#data resizing
def crop_resize_img(img):
    if img.shape[0] != img.shape[1]:
        img=img[0:120,10:150]
    resized_image = imresize(img, (rows,cols))
    return resized_image


# In[6]:


#using percentile to deal with outliers in the data.

def normalize_image(img):
    normalized_image= img - np.percentile(img,15)/ np.percentile(img,85) - np.percentile(img,15)
    return normalized_image


# In[7]:


def fetch_batchdata(source_path, folder_list, batch, batch_size, t):
    batch_data = np.zeros((batch_size,frames,rows,cols,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
    batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
    img_idx = [x for x in range(0, 30,2)] 
    for folder in range(batch_size): # iterate over the batch_size
        imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
        for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
            image = cv2.imread(source_path+'/'+ t[folder + (batch_num*batch_size)].strip().split(';')[0]+'/'+imgs[item], cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #crop the images and resize them. Note that the images are of 2 different shape 
            #and the conv3D will throw error if the inputs in a batch have different shapes
            resized_image=crop_resize_img(image)
                   
            batch_data[folder,idx,:,:,0] = normalize_image(resized_image[:, : , 0])#normalise and feed in the image
            batch_data[folder,idx,:,:,1] = normalize_image(resized_image[:, : , 1])#normalise and feed in the image
            batch_data[folder,idx,:,:,2] = normalize_image(resized_image[:, : , 2])#normalise and feed in the image
                    
            batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
    return batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do
    


# In[8]:


def fetch_aug_batchdata(source_path, folder_list, batch_num, batch_size, t,validation):
    
    # intialize variables to store data read from train data
    batch_data = np.zeros((batch_size,frames,rows,cols,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
    batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
    #print(batch_data)
    #print(batch_labels)
    # intialize variables for augumented batch data with affine transformation
    batch_data_aug,batch_label_aug = batch_data,batch_labels
    #print(batch_data_aug)
    #print(batch_label_aug)
    # intialize variables for augmented batch data with horizontal flip
    batch_data_flip,batch_label_flip = batch_data,batch_labels
    
    #create a list of image numbers you want to use for a particular video using full frames
    img_idx = [x for x in range(0, 30,2)] 
    
    for folder in range(batch_size): # iterate over the batch_size
        # read all the images in the folder
        imgs = sorted(os.listdir(source_path+'/'+ t[folder + (batch_num*batch_size)].split(';')[0])) 
        # create a random affine to be used in image transformation for buidling agumented data set
        dx, dy = np.random.randint(-1.7, 1.8, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        #  Iterate over the frames for each folder to read them in
        for idx, item in enumerate(img_idx):             
            image = cv2.imread(source_path+'/'+ t[folder + (batch_num*batch_size)].strip().split(';')[0]+'/'+imgs[item], cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)           
            # Cropping non symmetric frames
            #crop the images and resize them. Note that the images are of 2 different shape 
            #and the conv3D will throw error if the inputs in a batch have different shapes 
            resized_image=crop_resize_img(image)
            
            #Normal data
            batch_data[folder,idx,:,:,0] = normalize_image(resized_image[:, : , 0])#normalise and feed in the image
            batch_data[folder,idx,:,:,1] = normalize_image(resized_image[:, : , 1])#normalise and feed in the image
            batch_data[folder,idx,:,:,2] = normalize_image(resized_image[:, : , 2])#normalise and feed in the image
            
            x =resized_image.shape[0]
            y =resized_image.shape[1]
            #Data with affine transformation
            batch_data_aug[folder,idx] = (cv2.warpAffine(resized_image, M, (x,y)))
            
            # Data with horizontal flip
            batch_data_flip[folder,idx]= np.flip(resized_image,1)

        batch_labels[folder, int(t[folder + (batch_num*batch_size)].strip().split(';')[2])] = 1
        batch_label_aug[folder, int(t[folder + (batch_num*batch_size)].strip().split(';')[2])] = 1
        
       if int(t[folder + (batch_num * batch_size)].strip().split(';')[2])==0:
            batch_label_flip[folder, 1] = 1
        elif int(t[folder + (batch_num*batch_size)].strip().split(';')[2])==1:
            batch_label_flip[folder, 0] = 1                    
        else:
            batch_label_flip[folder, int(t[folder + (batch_num*batch_size)].strip().split(';')[2])] = 1
                  
    #adding the augumented data in the main data.
    
    batch_data_final = np.append(batch_data, batch_data_aug, axis = 0)
    batch_data_final = np.append(batch_data_final, batch_data_flip, axis = 0)

    batch_label_final = np.append(batch_labels, batch_label_aug, axis = 0) 
    batch_label_final = np.append(batch_label_final, batch_label_flip, axis = 0)
    
    if validation:
        batch_data_final=batch_data
        batch_label_final= batch_labels
        
    return batch_data_final,batch_label_final


# ## Generator
# This is one of the most important part of the code. The overall structure of the generator has been given. In the generator, you are going to preprocess the images as you have images of 2 different dimensions as well as create a batch of video frames. You have to experiment with `img_idx`, `y`,`z` and normalization such that you get high accuracy.

# In[10]:


def generator(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [x for x in range(0, 30,2)] #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)//batch_size # calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            yield fetch_batchdata(source_path, folder_list, batch, batch_size, t)

        
        # write the code for the remaining data points which are left after full batches
        if (len(folder_list) != batch_size*num_batches):
            batch_size = len(folder_list) - (batch_size*num_batches)
            yield fetch_batchdata(source_path, folder_list, batch, batch_size, t)


# In[11]:


def aug_generator(source_path, folder_list, batch_size, validation=False,ablation=None):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    if(ablation!=None):
        folder_list=folder_list[:ablation]
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)//batch_size # calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            # you yield the batch_data and the batch_labels, remember what does yield do
            yield fetch_aug_batchdata(source_path, folder_list, batch, batch_size, t,validation)
            
        
        # Code for the remaining data points which are left after full batches
        if (len(folder_list) != batch_size*num_batches):
            batch_size = len(folder_list) - (batch_size*num_batches)
            yield fetch_aug_batchdata(source_path, folder_list, batch, batch_size, t,validation)


# Note here that a video is represented above in the generator as (number of images, height, width, number of channels). Take this into consideration while creating the model architecture.

# In[12]:


curr_dt_time = datetime.datetime.now()
train_path = 'Project_data/train'
val_path = 'Project_data/val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = 25# choose the number of epochs
print ('# epochs =', num_epochs)


# ## Model
# Here you make the model using different functionalities that Keras provides. Remember to use `Conv3D` and `MaxPooling3D` and not `Conv2D` and `Maxpooling2D` for a 3D convolution model. You would want to use `TimeDistributed` while building a Conv2D + RNN model. Also remember that the last layer is the softmax. Design the network in such a way that the model is able to give good accuracy on the least number of parameters so that it can fit in the memory of the webcam.

# In[13]:


from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout,LSTM
from keras.layers.convolutional import Conv3D,Conv2D, MaxPooling3D,MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers.recurrent import LSTM, GRU

#write your model here
nb_featuremap = [8,16,32,64]
nb_dense = [128,64,5]

# Input
input_shape=(frames,rows,cols,channel)

# Define model
model = Sequential()

##Conv3D layers
#-----------------------------------------------------------

model.add(Conv3D(nb_featuremap[0], 
                 kernel_size=(5,5,5),
                 input_shape=input_shape,
                 padding='same', name="conv1"))
model.add(Activation('relu'))


model.add(Conv3D(nb_featuremap[1], 
                 kernel_size=(3,3,3),
                 padding='same',name="conv2"))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))


model.add(Conv3D(nb_featuremap[2], 
                 kernel_size=(1,3,3), 
                 padding='same',name="conv3"))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))

model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling3D(pool_size=(2,2,2)))

model.add(Flatten())

model.add(Dense(nb_dense[0], activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_dense[1], activation='relu'))

#softmax layer
model.add(Dense(nb_dense[2], activation='softmax'))


# ### Now... adding the CONV2D models along with Time Distributed layers to the model..

# In[14]:


nb_classes = 5
nb_featuremap = [8,16,32,64,128,256]
nb_dense = [128,64,5]

model = Sequential()

model.add(TimeDistributed(Conv2D(nb_featuremap[0], (3, 3), strides=(2, 2),activation='relu', padding='same'), input_shape=input_shape))


model.add(TimeDistributed(Conv2D(nb_featuremap[1], (3,3),padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(nb_featuremap[2], (3,3),padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(Conv2D(nb_featuremap[3], (2,2),padding='same', activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

model.add(TimeDistributed(BatchNormalization()))
model.add(Dropout(0.25))

model.add(TimeDistributed(Flatten()))

model.add(Dense(nb_dense[0], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_dense[1], activation='relu'))
model.add(Dropout(0.25))

## using GRU as the RNN model along with softmax as our last layer.
model.add(GRU(128, return_sequences=False))
model.add(Dense(nb_classes, activation='softmax'))


# Now that you have written the model, the next step is to `compile` the model. When you print the `summary` of the model, you'll see the total number of parameters you have to train.

# In[15]:


from keras.optimizers import Adam

optimiser = Adam()#write your optimizer
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model.summary())


# Let us create the `train_generator` and the `val_generator` which will be used in `.fit_generator`.

# In[16]:


train_generator = aug_generator(train_path, train_doc, batch_size)
val_generator = aug_generator(train_path, train_doc, batch_size)


# In[17]:


model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)
        
filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown=4, verbose=1,mode='auto',epsilon=0.0001)# write the REducelronplateau code here
callbacks_list = [checkpoint, LR]


# The `steps_per_epoch` and `validation_steps` are used by `fit_generator` to decide the number of next() calls it need to make.

# In[18]:


if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1


# Let us now fit the model. This will start training the model and with the help of the checkpoints, you'll be able to save the model at the end of each epoch.

# In[19]:


model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1,callbacks=callbacks_list, validation_data=val_generator,validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0,use_multiprocessing=True)

