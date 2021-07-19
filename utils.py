import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import apply_affine_transform
from tensorflow.image import flip_left_right, flip_up_down
from sklearn.metrics import confusion_matrix

def norm_brain(X):
    ''' Normalize the brain area of the input image X.
    '''
    Nchannels = X.shape[-1]
      
    X_norm = np.zeros_like(X)
    for channel in range(Nchannels): 
        im = X[:,:,channel]
        brain = im[im!=0]
        
        ########## YOUR CODE ############
        # apply normalization on the brain area
        
        
        ##########    END    ############
        
        im[im!=0] = brain_norm
        X_norm[:,:,channel] = im
                      
        del im, brain, brain_norm
    
    return X_norm

def load_img(filename, Nclasses, norm=False):
    ''' Load one image and its true segmentation
    '''
    images = np.load(filename)    
    X = images[:, :, 0:4]
    
    # perform normalization
    if norm == True:
        X = norm_brain(X)
    
    y = images[:, :, -1]
    
    if Nclasses==2:
        y[y>0] = 1
    
    return X, y
        
def visualize(img, seg):
    ''' Plot the 4 differen MRI modalities and its segmentation
    '''
    if len(seg.shape)>2:
        seg = np.argmax(seg, axis=-1)
    
    mri_names = ['T1', 'T2', 'T1ce', 'FLAIR']
    
    plt.figure(figsize=(15,15))
    for i in range(4):
        plt.subplot(151+i)
        im = img[:,:,i]
        im = (im-np.amin(im))/(np.amax(im)-np.amin(im))
        
        plt.imshow(im, cmap='gray')
        plt.title(mri_names[i])
        plt.axis('off')
        
    plt.subplot(155)
    plt.imshow(seg, cmap='gray')
    plt.title('Seg')
    plt.axis('off')
    plt.show()
    
def plot_trends(results):
    # Accuracy trend
    plt.figure(figsize=(10,4))
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='lower right')

    # Dice trend
    plt.figure(figsize=(10,4))
    plt.plot(results.history['dice'])
    plt.plot(results.history['val_dice'])
    plt.ylabel('Dice')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='lower right')

    # Loss trend
    plt.figure(figsize=(10,4))
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.show()
    
def augmentation(X, y, do):
    ''' Apply image augmentation (rotation, translation, shear, zoom, and flip) on the image X and its segmentation y.
    '''
    # lets garantee to use at least 25% of original images
    if np.random.random_sample()>0.75:
        return X, y
    
    else:
        # affine transformation default parameters
        alpha, Tx, Ty, beta, Zx, Zy = 0, 0, 0, 0, 1, 1
        
        ########## YOUR CODE ############
        # implement ROTATION with random angle alpha between [-45, +45) degrees
        if do[0] == 1:   
            alpha = 0

        # implement TRANSLATION with random values Tx, Ty between 0 and the 10% of the image shape
        if do[1] == 1: 
            Tx, Ty = 0, 0

        # implement SHEAR with random angle beta between [0-10) degrees
        if do[2] == 1: 
            beta = 0

        # implement random ZOOM Zx, Zy between -20%, +20% along the 2 axis
        if do[3] == 1: 
            Zx, Zy = 1, 1
        
        ##########    END    ############
        
        
        # apply affine tranformation to the image
        X_new = apply_affine_transform(X, 
                                      theta=alpha,      # rotation
                                      tx=Tx, ty=Ty,     # translation
                                      shear=beta,       # shear
                                      zx=Zx, zy=Zy,     # zoom
                                      row_axis=0, col_axis=1, channel_axis=2, 
                                      fill_mode='constant', cval=0.0, 
                                      order=1)

        # apply affine tranformation to the target
        y_new = apply_affine_transform(y, 
                                      theta=alpha,      # rotation
                                      tx=Tx, ty=Ty,     # translation
                                      shear=beta,       # shear
                                      zx=Zx, zy=Zy,     # zoom
                                      row_axis=0, col_axis=1, channel_axis=2, 
                                      fill_mode='constant', cval=0.0, 
                                      order=0)
        
        
        # FLIPPING
        if do[4] == 1:
            choice = np.random.randint(3)

            # left-right flipping
            if choice == 0:
                X_new, y_new = flip_left_right(X_new), flip_left_right(y_new)

            # up-down flipping    
            if choice == 1:
                X_new, y_new = flip_up_down(X_new), flip_up_down(y_new)

            # both flipping
            if choice == 2:
                X_new, y_new = flip_left_right(X_new), flip_left_right(y_new)
                X_new, y_new = flip_up_down(X_new), flip_up_down(y_new)

            X_new, y_new = X_new.numpy(), y_new.numpy()

        return X_new, y_new

def aug_batch(Xb, yb):
    ''' Generate a augmented image batch 
    '''
    batch_size = len(Xb)
    Xb_new, yb_new = np.empty_like(Xb), np.empty_like(yb)
    
    for i in range(batch_size):
        decisions = np.random.randint(2, size=5) # 5 is number of augmentation techniques to combine
        X_aug, y_aug = augmentation(Xb[i], yb[i], decisions)
        Xb_new[i], yb_new[i] = X_aug, y_aug
        
    return Xb_new, yb_new      
    
class DataGenerator(tf.keras.utils.Sequence):
    ''' Keras Data Generator
    '''
    def __init__(self, list_IDs, n_classes, batch_size=8, dim=(160,192), n_channels=4, norm=False, augmentation=False):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.norm = norm
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        ''' Denotes the number of batches per epoch
        '''
        return len(self.list_IDs)//self.batch_size

    def __getitem__(self, index):
        ''' Generate one batch of data
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data     
        X, y = self.__data_generation(list_IDs_temp)
        if self.augmentation == True:
            X, y = self.__data_augmentation(X, y)
        
        if index == self.__len__()-1:
            self.on_epoch_end()
        
        return X, y

    def on_epoch_end(self):
        ''' Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            # Store sample
            X[i], y[i] = load_img(IDs, self.n_classes, self.norm)
            
        return X.astype('float32'), to_categorical(y, self.n_classes)

    def __data_augmentation(self, X, y):
        'Apply augmentation'
        X_aug, y_aug = aug_batch(X, y)
        
        return X_aug, y_aug