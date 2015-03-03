"""
National Data Science Bowl:
Predict ocean health, one plankton at a time
author: Elena Cuoco
date: 18/01/2015


    1. prepare the images and put them in a csv files on disk once for all
    2. denoise image, find countor plot
    3. extract features find the min-max,peak to add to set
    4.http://scikit-image.org/docs/dev/auto_examples/

"""
#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
import glob
import os
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import  filter
from skimage.morphology import medial_axis
from skimage.filter.rank import entropy
from skimage.morphology import disk
from skimage.exposure import equalize_hist
from skimage.morphology import skeletonize
from skimage.feature import ORB
from sklearn.feature_extraction import image as s_im
from sklearn.cluster import spectral_clustering
from skimage.feature import hog
from skimage.filter import   threshold_adaptive
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.filter import threshold_otsu
import mahotas
import mahotas.features

################# useful functions#############################################################

def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)

    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio
# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def disk_structure(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.bool)


def granulometry(data, sizes=None):
    s = max(data.shape)
    if sizes == None:
        sizes = range(1, s/2, 2)
    granulo = [ndimage.binary_opening(data, \
            structure=disk_structure(n)).sum() for n in sizes]
    return granulo

def image_multi_features(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric
     mask = img > img.mean()
     label_im, nb_labels = ndimage.label(mask)
     X=np.zeros(num_features, dtype=float)
     # image=ndimage.median_filter(img, 3)
     X[imageSize]=img.max()
     X[imageSize+1]=img.min()
     X[imageSize+2]=nb_labels
     extra_sizes=[4,8,12,16,20,24]


     image = resize(img, (maxPixel, maxPixel))
     granulo = granulometry(image,sizes=extra_sizes)

     sx = ndimage.sobel(image, axis=0, mode='constant')
     sy = ndimage.sobel(image, axis=1, mode='constant')
     sob = np.hypot(sx, sy)
     #edges=canny(image,3,0.3,0.2)
     # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(sob,(1, imageSize))
     for i in range(len(extra_sizes)):
      X[imageSize+3+i]=granulo[i]

     return X

def image_features_sobel(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=np.float32)

     image = filter.sobel(img)
     edges = resize(image, (maxPixel, maxPixel))
    # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(edges,(1, imageSize))

     return X

def image_features_morphology(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=float)
     image = resize(img, (maxPixel, maxPixel))

     # Compute the medial axis (skeleton) and the distance transform
     skel, distance = medial_axis(image, return_distance=True)

     # Distance to the background for pixels of the skeleton
     dist_on_skel = distance * skel

    # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(dist_on_skel,(1, imageSize))

     return X
def image_features_entropy(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=float)
     image = resize(img, (maxPixel, maxPixel))
     imag=entropy(image, disk(5))

    # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(imag,(1, imageSize))

     return X
def image_features_equalize(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=float)
     image = resize(img, (maxPixel, maxPixel))

     imag= equalize_hist(image)
    # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(imag,(1, imageSize))

     return X
def image_features_equalize_binary(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=float)
     image = resize(img, (maxPixel, maxPixel))

     imag= equalize_hist(image)
     im = np.where(imag > np.mean(imag),1.0,0.0)

    # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(im,(1, imageSize))

     return X

def image_features_binary(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=float)
     image = resize(img, (maxPixel, maxPixel))
     imag = np.where(image > np.mean(image),1.0,0.0)

    # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(imag,(1, imageSize))

     return X
def image_features_skeletonize(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pi
     X=np.zeros(num_features, dtype=float)
     image = resize(img, (maxPixel, maxPixel))
     binary_image = np.where(image < np.mean(image),1.0,0.0)
     imag = skeletonize(binary_image)



    # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(imag,(1, imageSize))

     return X
def image_features_resize_adaptive(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric
     block_size = 20
     im = threshold_adaptive(img, block_size, offset=5)
     X=np.zeros(num_features, dtype=float)
     image = resize(im, (maxPixel, maxPixel))
     # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(image,(1, imageSize))

     return X
def image_features_resize(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=float)
     image = resize(img, (maxPixel, maxPixel))
     # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(image,(1, imageSize))

     return X
def image_features_resize_thres(img, maxPixel, num_features,imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric

     X=np.zeros(num_features, dtype=float)
     image=denoise_bilateral(img, win_size=5, sigma_range=None, sigma_spatial=1, bins=10000, mode='constant', cval=0)
     thresh = threshold_otsu(image)
     binary = image > thresh
     im = resize(binary, (maxPixel, maxPixel))

     # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(im,(1, imageSize))

     return X

def image_features_patches(img,p,max_patches):
     # X is the feature vector with one row of features per image
     #
     Xsize=p*p*max_patches
     X=np.zeros(Xsize, dtype=float)

     # extract patches using scikit library.
     patches = s_im.extract_patches_2d(img, (p,p), max_patches=max_patches,random_state=0)
     X[0:Xsize] = np.reshape(patches,(1, Xsize))

     return X
def image_features_hog(img, num_features,orientation,maxcell,maxPixel):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric
     imag= equalize_hist(img)
     image = np.where(imag > np.mean(imag),1.0,0.0)
     im = resize(image, (maxPixel, maxPixel))
     ##hog scikit transform
     fd= hog(im, orientations=orientation, pixels_per_cell=(maxcell, maxcell),
                    cells_per_block=(1, 1), visualise=False,normalise=True)

     return fd
def image_features_hog2(img, num_features,orientation,maxcell,maxPixel):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric
     block_size = 10
     image = threshold_adaptive(img, block_size, offset=5)
     im = resize(image, (maxPixel, maxPixel))
     ##hog scikit transform
     fd= hog(im, orientations=orientation, pixels_per_cell=(maxcell, maxcell),
                    cells_per_block=(1, 1), visualise=False,normalise=True)

     return fd

def image_features_hog3(img, num_features,orientation,maxcell,maxPixel):

     image=denoise_bilateral(img, win_size=5, sigma_range=None, sigma_spatial=1, bins=10000, mode='constant', cval=0)
     thresh = threshold_otsu(image)
     binary = image > thresh
     im = resize(binary, (maxPixel, maxPixel))
     ##hog scikit transform
     fd= hog(im, orientations=orientation, pixels_per_cell=(maxcell, maxcell),
                    cells_per_block=(1, 1), visualise=False,normalise=True)

     return fd
def image_features_orb(img,keypoints):
     # X is the feature vector with one row of features per image
     #
     Xsize=2*keypoints
     X=np.zeros(Xsize, dtype=float)
     # extract patches using scikit library.
     orb=ORB(downscale=1.2, n_scales=8, n_keypoints=keypoints, fast_n=4, fast_threshold=0.00001, harris_k=0.01)
     orb.detect_and_extract(img)
     X[0:Xsize] = np.reshape(orb.keypoints,(1, Xsize))
     return X

def image_features_labels(img,n_clusters,maxPixel):
     # X is the feature vector with one row of features per image
     #
     imageSize=maxPixel*maxPixel
     img = resize(img, (maxPixel, maxPixel))
     mask = img.astype(bool)
     # Convert the image into a graph with the value of the gradient on the
     # edges.
     graph = s_im.img_to_graph(img, mask=mask)

     # Take a decreasing function of the gradient: we take it weakly
     # dependent from the gradient the segmentation is close to a voronoi
     graph.data = np.exp(-graph.data / graph.data.std())

     # Force the solver to be arpack, since amg is numerically
     # unstable on this example
     labels = spectral_clustering(graph, n_clusters, eigen_solver='arpack')
     label_im = -np.ones(mask.shape)
     label_im[mask] = labels

     X=np.zeros(imageSize, dtype=float)

     # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(label_im,(1, imageSize))
     return X
def image_features_haralick(img, imageSize):
     # X is the feature vector with one row of features per image
     #  consisting of the pixel values a, num_featuresnd our metric
     feats=mahotas.features.haralick(img, ignore_zeros=False, preserve_haralick_bug=False, compute_14th_feature=False)
     # Store the rescaled image pixels
     X[0:imageSize] = np.reshape(feats,(1, imageSize))

     return X
################
###################################End of useful functions#########################################
#
################Start Features extraction and data preparation#####################################
direc='./data'
#direc='/users/cuoco/home/workspace/git/bowl/data'
# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join(direc,"train", "*"))\
 ).difference(set(glob.glob(os.path.join(direc,"train","*.*")))))
directory_names.sort()###important

############Parameters######################################
#######################
#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1
num_rows = numberofImages # one row for each image in the training dataset
# We'll rescale the images to the minimun size of all images
# minimum dimension is 21
maxPixel=64
imageSize = maxPixel * maxPixel
#imageSize=4*13##features.haralick
num_features = imageSize
###HOG Parameters

#orientation=11#number of orientation
#maxcell=4#minimun size per cell of HOG transform
#num_features = orientation * (maxPixel // maxcell) * (maxPixel // maxcell)
#imageSize=num_features
print imageSize
#############################################################
# X is the feature vector with one row of features per image
# consisting of the pixel values and other features
X = np.zeros((num_features), dtype=np.float32)
# Generate training data
i = 0
label = 0
##############TRAINING FILES
trainfile=direc+'/transformed/train.csv'
train = open(trainfile, 'wb')
header='image,label'
for i in range(imageSize):
 header+=','+str(i)
header+='\n'
train.write(header)
print "Reading training images"
# Navigate through the list of directories
namesClasses = list()
#labels_dict={}
for folder in directory_names:

 # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
            img =imread(nameFileImage,as_grey=True)
            Xar=str(fileName)+','+str(label)
            X= image_features_sobel(img, maxPixel, num_features,imageSize)

            for k in range(imageSize):
                Xar+=','+str(X[k])
            Xar+='\n'
            train.write(Xar)
            i += 1
            # report progress for each 5% done
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    #labels_dict[label]=currentClass        
    label += 1
train.close()
classes = map(lambda fileName: fileName.split('/')[-1], namesClasses)
print classes
#print labels_dict
print('saved train')


#######################----------TEST IMAGES----------##########################################
testfile=direc+'/transformed/test.csv'
test = open(testfile, 'wb')
header_test='image'
for i in range(imageSize):
 header_test+=','+str(i)
header_test+='\n'
test.write(header_test)

print "Reading test images"
#get the total test images
fnames = glob.glob(os.path.join(direc, "test", "*.jpg"))
numberofTestImages = len(fnames)
print numberofTestImages
images = map(lambda fileName: fileName.split('/')[-1], fnames)
X_test = np.zeros((num_features), dtype=np.float32)
i = 0
# report progress for each 5% done
report = [int((j+1)*numberofTestImages/20.) for j in range(20)]
for fileName in images:
    # Read in the images and create the features
    img = imread(direc+"/test/"+fileName,as_grey=True)

    Xar=str(fileName)
    X_test= image_features_sobel(img, maxPixel, num_features,imageSize)

    for k in range(imageSize):
        Xar+=','+str(X_test[k])
    Xar+='\n'
    test.write(Xar)
    i += 1
    if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"

test.close()
print('saved test')
