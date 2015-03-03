"""
National Data Science Bowl:
Predict ocean health, one plankton at a time
author: Elena Cuoco
Using NN and theano

credits:http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

"""
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn import preprocessing
#joblib library for serialization
from sklearn.externals import joblib
from scipy import misc
from sklearn import neural_network
import theano
import sys,os,shutil
from scipy import ndimage
from sklearn.utils import shuffle
from skimage import transform as tf
from skimage.transform import rotate
#json library for settings file
import json
import warnings
warnings.filterwarnings("ignore")
def nudge_dataset(X, Y,maxPixel=64):
    """
    This produces a dataset 8 times bigger than the original one,
    by rotate the maxPixel x maxPixel images
    """
    
    rotateImage= lambda x, w: rotate(x.reshape((maxPixel, maxPixel)),w,resize=False).ravel()
    #############rotate image without reshaping###############
    angles=[90,180]
    X = np.concatenate([X] +[np.apply_along_axis(rotateImage, 1, X, angle) for angle in angles])
    Y = np.concatenate([Y for _ in range(len(angles)+1)], axis=0)

    ###############equalize###################################
    ##############
    X=np.asarray(X.astype(np.float32))
    Y = np.asarray(Y,dtype=np.int32)
    return X, Y
def clean_data(data):
    y_train=data['label']
    clean=data.drop(['image','label'], axis=1)#remove image and label columns
    X_train=np.asarray(clean.astype(np.float32))
    y_train = np.asarray(y_train,dtype=np.int32)
    return y_train,X_train
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

#------------------NN-------------------------############
from lasagne import layers
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import identity
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import negative_log_likelihood
try:
 from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
 from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
 Conv2DLayer = layers.Conv2DLayer
 MaxPool2DL = layers.MaxPool2DLayer

Maxout = layers.pool.FeaturePoolLayer
maxPixel=64

net1 = NeuralNet(
layers=[
('input', layers.InputLayer),
('conv1', Conv2DLayer),
('pool1', MaxPool2DLayer),
('dropout1', layers.DropoutLayer),
('conv2', Conv2DLayer),
('pool2', MaxPool2DLayer),
('dropout2', layers.DropoutLayer),
('conv3', Conv2DLayer),
('pool3', MaxPool2DLayer),
('dropout3', layers.DropoutLayer),
('hidden4', layers.DenseLayer),
('maxout4', Maxout),
('dropout4', layers.DropoutLayer),
('hidden5', layers.DenseLayer),
('maxout5', Maxout),
('dropout5', layers.DropoutLayer),
('output', layers.DenseLayer),

],
    input_shape=(None, 1, maxPixel, maxPixel),
conv1_num_filters=32, conv1_filter_size=(4, 4), pool1_ds=(2, 2),
dropout1_p=0.2,
conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_ds=(2, 2),
dropout2_p=0.2,
conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
dropout3_p=0.2,
hidden4_num_units=1024,
dropout4_p=0.3,
maxout4_ds=2,
hidden5_num_units=2048,
dropout5_p=0.5,
maxout5_ds=2,
output_num_units=121,
output_nonlinearity=softmax,
update_learning_rate=theano.shared(np.float32(0.03)),
update_momentum=theano.shared(np.float32(0.9)),
regression=False,
#batch_iterator_train=DataAugmentationBatchIterator(batch_size=128),
on_epoch_finished=[
AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
AdjustVariable('update_momentum', start=0.9, stop=0.999),

EarlyStopping(patience=20),
],
max_epochs=3000,
verbose=1,
eval_size=0.2
)



#-------------------------------------------###############
##Read configuration parameters
file_dir = './settings-nb.json'
config = json.loads(open(file_dir).read())
train_file=config["HOME"]+config["TRAIN_DATA_PATH"]+'train.csv'

MODEL_PATH=config["HOME"]+config["MODEL_PATH"]
seed= int(config["SEED"])
path_to_delete =MODEL_PATH
if os.path.exists(path_to_delete):
    shutil.rmtree(path_to_delete)
os.mkdir(MODEL_PATH)
###################################################################################
# start training
start = datetime.now()

if __name__=='__main__':
 # Load Data
 data= pd.read_csv(train_file,index_col=False)
 yn,Xn=clean_data(data)


 X,y=nudge_dataset(Xn, yn,maxPixel)
 X, y = shuffle(X, y, random_state=seed)
 print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
 print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max())) 
 ###prepare train and  test sets
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2,random_state=seed)
 #X_train = X_train.reshape(-1, 1, maxPixel,maxPixel)
 #X_test = X_test.reshape(-1, 1, maxPixel,maxPixel)
 X = X.reshape(-1, 1, maxPixel,maxPixel)

 net1.fit(X, y)
 #serialize training
 model_file=MODEL_PATH+'model-clf.pkl'
 joblib.dump(net1, model_file)
 #y_pred=net1.predict_proba(X_test)
 #MCL=log_loss(y_test, y_pred)
 #print('MLC Results(scikit version) :'),
 #print MCL
 print('elapsed time: %s ' % (str(datetime.now() - start)))
