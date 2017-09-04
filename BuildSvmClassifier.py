import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV

#from sklearn.externals import joblib # to unmarshalle the classifier & scaler
import SaveAndRestoreClassifier

import sklearn
sklearnVersion=[int(i)  for i in sklearn.__version__.split(".")]

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split

sklearnVersion18=sklearnVersion[1] >= 18
print("sklearn._version_:", sklearn.__version__, ", sklearnVersion:", sklearnVersion, ", sklearnVersion18?", sklearnVersion18)

if sklearnVersion18:
    from sklearn.model_selection import train_test_split
    print("from sklearn.model_selection import train_test_split")
else:
    from sklearn.cross_validation import train_test_split
    print("from sklearn.cross_validation import train_test_split")


# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

TESTING=False
USESMALLSET=False
USESMALLSSAMPLESIZE=True
LINEAR=True
'''

LINEAR=True

Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
svmGridSearch-theParameters: [{'C': [0.0001, 0.0003, 0.0004, 0.0005], 'kernel': ['linear']}]
svmGridSearch-Best parameters set found on development set: {'kernel': 'linear', 'C': 0.0001} , best score: 0.990427927928

svmGridSearch-Grid scores on development set:

0.990 (+/-0.002) for {'kernel': 'linear', 'C': 0.0001}
0.990 (+/-0.001) for {'kernel': 'linear', 'C': 0.0003}
0.990 (+/-0.002) for {'kernel': 'linear', 'C': 0.0004}
0.990 (+/-0.002) for {'kernel': 'linear', 'C': 0.0005}
gridSearchSvc.best_estimator_: SVC(C=0.0001, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False) , gridSearchSvc: GridSearchCV(cv=5, error_score='raise',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid=[{'C': [0.0001, 0.0003, 0.0004, 0.0005], 'kernel': ['linear']}],
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring=None, verbose=0)  in: 3366.52  seconds
0.0 Seconds to train SVC...on: 14208  samples
Training Accuracy of SVC =  0.9955
Test Accuracy of SVC =  0.9913
My SVC predicts:  [ 1.  0.  1. ...,  0.  0.  0.]
For these 3552 labels:  [ 1.  0.  1. ...,  0.  0.  0.]
30.03418 Seconds to predict 3552 labels with SVC
saving model to:  ./models/HogClassify.svm
saveClassifier-saving model to: <./models/HogClassify.svm>
restoring model from:  ./models/HogClassify.svm
restoreClassifier-restoring model from: <./models/HogClassify.svm>
My restored SVM classifier predicts:  [ 1.  0.  1. ...,  0.  0.  0.]
For these 3552 labels:  [ 1.  0.  1. ...,  0.  0.  0.]
Test Accuracy of restored classifier =  0.9913
90.2915 Seconds to predict 3552 labels with SVC



'''
MODELFILENAME='./models/HogClassify.svm'
SCALERFILENAME='./models/HogClassifyScaler.svm'

'''
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

'''
import BuildFeatureVector

# Divide up into vehicleImageNames and notvehicleImageNames
def collectImageNames(theDirectories):
    allImageNames=[]
    for directory in theDirectories:
        pngImageNamePattern=directory+'/*.png'
        jpgImageNamePattern=directory+'/*.jpeg'
        #print("directory:", directory, ", imageNamePattern:", imageNamePattern)
        imageNames=glob.glob(pngImageNamePattern) # png
        #print("len(imageNames):", len(imageNames))
        allImageNames+=imageNames
        imageNames=glob.glob(jpgImageNamePattern) # jpeg
        #print("len(imageNames):", len(imageNames))
        allImageNames+=imageNames
    #print("len(allImageNames):", len(allImageNames))
    return allImageNames;

TRAININGDIRECTORY="./training/"
SMALLSETDIRECTORY=TRAININGDIRECTORY+"smallset/"
VEHICLETRAININGDIRECTORY=TRAININGDIRECTORY+"vehicles/"
if USESMALLSET:
    VEHICLETRAININGDIRECTORY=SMALLSETDIRECTORY+"vehicles_smallset/"
vehicleDirectories=glob.glob(VEHICLETRAININGDIRECTORY+'*')
vehicleImageNames=collectImageNames(vehicleDirectories)
print("len(vehicleImageNames):", len(vehicleImageNames), ", vehicleDirectories:", vehicleDirectories, ", VEHICLETRAININGDIRECTORY:", VEHICLETRAININGDIRECTORY)

NONVEHICLETRAININGDIRECTORY=TRAININGDIRECTORY+"non-vehicles/"
if USESMALLSET:
    NONVEHICLETRAININGDIRECTORY=SMALLSETDIRECTORY+"non-vehicles_smallset/"
nonvehicleDirectories=glob.glob(NONVEHICLETRAININGDIRECTORY+'*')
nonvehicleImageNames=collectImageNames(nonvehicleDirectories)
print("len(nonvehicleImageNames):", len(nonvehicleImageNames), ", nonvehicleDirectories:", nonvehicleDirectories, ", NONVEHICLETRAININGDIRECTORY:", NONVEHICLETRAININGDIRECTORY)

if USESMALLSSAMPLESIZE:
    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    SAMPLESIZE = 500
    SAMPLESIZE = 5000
    vehicleImageNames=vehicleImageNames[0:SAMPLESIZE]
    print("len(vehicleImageNames):", len(vehicleImageNames))
    nonvehicleImageNames=nonvehicleImageNames[0:SAMPLESIZE]
    print("len(nonvehicleImageNames):", len(nonvehicleImageNames))

### TODO: Tweak these parameters and see how the results change.
#COLORSPACE = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#orient = 9
#pix_per_cell = 8
#cell_per_block = 2
#hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
import FeatureVectorConfig
t=time.time()
'''
car_features = extract_features(vehicleImageNames, cspace=COLORSPACE, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features(nonvehicleImageNames, cspace=COLORSPACE, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
'''
car_features = BuildFeatureVector.extract_features(vehicleImageNames, color_space=FeatureVectorConfig.COLORSPACE, orient=FeatureVectorConfig.ORIENTATIONBINS, 
                        pix_per_cell=FeatureVectorConfig.PIXELSPERCELL, cell_per_block=FeatureVectorConfig.CELLSPERBLOCK, 
                        hog_channel=FeatureVectorConfig.HOGCHANNEL,
                        spatial_feat=FeatureVectorConfig.SPATIALFEATURES,
                        spatial_size=FeatureVectorConfig.SPATIALSIZE,
                        hist_feat=FeatureVectorConfig.HISTOGRAMFEATURES,
                        hist_bins=FeatureVectorConfig.HISTOGRAMBINS,
                        hog_feat=FeatureVectorConfig.HOGFEATURES)
notcar_features = BuildFeatureVector.extract_features(nonvehicleImageNames, color_space=FeatureVectorConfig.COLORSPACE, orient=FeatureVectorConfig.ORIENTATIONBINS, 
                        pix_per_cell=FeatureVectorConfig.PIXELSPERCELL, cell_per_block=FeatureVectorConfig.CELLSPERBLOCK, 
                        hog_channel=FeatureVectorConfig.HOGCHANNEL,
                        spatial_feat=FeatureVectorConfig.SPATIALFEATURES,
                        spatial_size=FeatureVectorConfig.SPATIALSIZE,
                        hist_feat=FeatureVectorConfig.HISTOGRAMFEATURES,
                        hist_bins=FeatureVectorConfig.HISTOGRAMBINS,
                        hog_feat=FeatureVectorConfig.HOGFEATURES)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# save the scaler
print('X_scaler: ', X_scaler, ", get_params:", X_scaler.get_params(deep=True), ", mean:", X_scaler.mean_, ", std:", X_scaler.std_)
print('saving scaler to: ', SCALERFILENAME)
#SaveAndRestoreClassifier.saveScalerFitX(X, SCALERFILENAME)
SaveAndRestoreClassifier.saveScaler(X_scaler, SCALERFILENAME)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',FeatureVectorConfig.ORIENTATIONBINS,'orientations',FeatureVectorConfig.PIXELSPERCELL,
    'pixels per cell and', FeatureVectorConfig.CELLSPERBLOCK,'cells per block')
print('Feature vector length:', len(X_train[0]))

CROSSVALIDATION=5
def svmGridSearch(X_train, y_train, theParameters):
    print("svmGridSearch-theParameters:", theParameters)
    svr = svm.SVC()
    gridSearchSvc = GridSearchCV(svr, theParameters, cv=CROSSVALIDATION)
    gridSearchSvc.fit(X_train, y_train)
    if True:
        print("svmGridSearch-Best parameters set found on development set:", gridSearchSvc.best_params_, ", best score:", gridSearchSvc.best_score_)
        print()
        print("svmGridSearch-Grid scores on development set:")
        print()
        means = gridSearchSvc.cv_results_['mean_test_score']
        stds = gridSearchSvc.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gridSearchSvc.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    return gridSearchSvc

def svmRbfGridSearch(X_train, y_train):
    C_range = np.logspace(-2, 10, 13)
    C_range = np.logspace(3, 4, 2)
    gamma_range = np.logspace(-9, 3, 13)
    #C_range = np.logspace(-2, -3, 2)
    gamma_range = np.logspace(-5, -3, 3)
    parameters = dict(gamma=gamma_range, C=C_range)
    parameters['kernel']= ['rbf']
    return svmGridSearch(X_train, y_train, parameters)

def svmLinearGridSearch(X_train, y_train):
    parameters = [{'kernel': ['linear'], 'C': [.00001, .00005, .0001, .0005, .001, .01, .1, 1]}]
    parameters = [{'kernel': ['linear'], 'C': [.0001, .0003,.0004,.0005]}]
    return svmGridSearch(X_train, y_train, parameters)

t=time.time()
gridSearchSvc=None
if LINEAR:
    gridSearchSvc=svmLinearGridSearch(X_train, y_train)
else:
    gridSearchSvc=svmRbfGridSearch(X_train, y_train)
t2 = time.time()
print ("gridSearchSvc.best_estimator_:", gridSearchSvc.best_estimator_, ", gridSearchSvc:", gridSearchSvc, " in:", round(t2-t, 2), " seconds")
# Use a linear SVC 
#svc = LinearSVC(C=gridSearchParameters.best_estimator_.C) # (re)train
# Check the training time for the SVC
t=time.time()
#gridSearchSvc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...on:', len(y_train), " samples")
# Check the score of the SVC
print('Training Accuracy of SVC = ', round(gridSearchSvc.score(X_train, y_train), 4))
print('Test Accuracy of SVC = ', round(gridSearchSvc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

numberOfPredictions=len(y_test)
if TESTING:
    numberOfPredictions = 100
    X_test=X_test[0:numberOfPredictions]
    y_test=y_test[0:numberOfPredictions]

svcPredictions=gridSearchSvc.predict(X_test)
print('My SVC predicts: ', svcPredictions)
print('For these',numberOfPredictions, 'labels: ', y_test)
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', numberOfPredictions,'labels with SVC')

# save the model
print('saving model to: ', MODELFILENAME)
SaveAndRestoreClassifier.saveClassifier(gridSearchSvc, MODELFILENAME)

print('restoring model from: ', MODELFILENAME)
classifier= SaveAndRestoreClassifier.restoreClassifier(MODELFILENAME)
classifierPredictions=classifier.predict(X_test)
print('My restored SVM classifier predicts: ', classifierPredictions)
print('For these',numberOfPredictions, 'labels: ', y_test)
print('Test Accuracy of restored classifier = ', round(classifier.score(X_test, y_test), 4))
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', numberOfPredictions,'labels with SVC')
#print("svcPredictions-classifierPredictions:", (svcPredictions-classifierPredictions), ", not all:", not (svcPredictions-classifierPredictions).all())
assert not (svcPredictions-classifierPredictions).all() # subtract equals and assert not all


