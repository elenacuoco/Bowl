# Bowl
Bowl kaggle competition (More info on kaggle website http://www.kaggle.com/c/datasciencebowl)
Predict ocean health, one plankton at a time 

##Software:
Language:python
This code prepare the images for the classification problem, do training usign CNN with nolearn, theano, lasagne.
Credits to the posts on D. Nouri blog
http://danielnouri.org/notes/

There are 3 files:
###Prepare_Features.py:
Useful funtions for image preprocessing. The results of preprocessing and resizing are saved on disk as np.array with labels from 0 to 120. The encoded labels printed can be used for submission.
###Training-NET.py 
Build a Neural Net with different layers, based on Nouri setup, train the net and save the model on disk 
###Prediction-NET.py 
Load the saved model and perform prediction, writing them in a file  for submission.


You can use a json file to set the PATH and other parameters once for all.



####Other useful repository links
https://github.com/msegala/Kaggle-National_Data_Science_Bowl


