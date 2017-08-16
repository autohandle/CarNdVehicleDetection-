import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
#from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


import SaveAndRestoreClassifier#
#MODELFILENAME='./models/RbfClassifier.svm'
MODELFILENAME='./models/Linear.svm'
print('restoring model from: ', MODELFILENAME)
classifier = SaveAndRestoreClassifier.restoreClassifier(MODELFILENAME)


#SCALERFILENAME='./models/RbfScaler.svm'
SCALERFILENAME='./models/LinearScaler.svm'
print('restoring scaler from: ', SCALERFILENAME)
scaler= SaveAndRestoreClassifier.restoreScaler(SCALERFILENAME)
print('scaler: ', scaler, ", get_params:", scaler.get_params(deep=True),
    ", mean:", scaler.mean_, ", len(mean):", len(scaler.mean_),
    ", scale:", scaler.scale_, ", len(scale):", len(scaler.scale_))
scaler=None

from skimage.feature import hog
#from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
'''
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
'''
# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
import BuildFeatureVector
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    if scaler!=None:
        print('search_windows-scaler: ', scaler, ", get_params:", scaler.get_params(deep=True), ", mean:", scaler.mean_, ", std:", scaler.std_)
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = BuildFeatureVector.singleImageFeatures(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #print ("len(features):", len(features))
        #print("search_windows-scaler:", scaler)
        if (scaler==None):
            scaler = StandardScaler().fit(features.reshape(-1, 1)) # however many rows & 1 column, i.e. only rows, rows x 1
            print("search_windows-len(features):", len(features))
            #print('search_windows-making-scaler: ', scaler, ", get_params:", scaler.get_params(deep=True), ", mean:", scaler.mean_, ", scale:", scaler.scale_)
        #5) Scale extracted features to be fed to classifier
        #print("search_windows-len(features):", len(features), ", features.shape:", features.shape)
        test_features = scaler.transform(np.array(features).reshape(1, -1)) # convert features to 1xfeatures (samplesxfeatures) before transform
        #6) Predict using your classifier
        #print("search_windows-len(test_features):", len(test_features), ", test_features.shape:", test_features.shape)
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
'''
# Read in cars and notcars
images = glob.glob('*.jpeg')
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
'''
import FeatureVectorConfig

y_start_stop = [None, None] # Min and max in y to search in slide_window()
y_start_stop = [300, 600] # Min and max in y to search in slide_window()
#y_start_stop = [None, None] # Min and max in y to search in slide_window()
'''
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
'''
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0]==None:
        x_start_stop[0]=0
    if x_start_stop[1]==None:
        x_start_stop[1]=img.shape[1]
    if y_start_stop[0]==None:
        y_start_stop[0]=0
    if y_start_stop[1]==None:
        y_start_stop[1]=img.shape[0]
    print("x_start_stop:", x_start_stop, ", y_start_stop:", y_start_stop)
    # Compute the span of the region to be searched 
    span=(y_start_stop[1]-y_start_stop[0], x_start_stop[1]-x_start_stop[0])
    # Compute the number of pixels per step in x/y
    pixelsPerStep=(int(round(xy_window[0]*(1.-xy_overlap[0])+0.5)),int(round(xy_window[1]*(1.-xy_overlap[1])+0.5)))
    print("span:", span, ", pixelsPerStep:", pixelsPerStep)
    # Compute the number of windows in x/y
    numberOfSteps=(int(span[0]/pixelsPerStep[0])-1,int(span[1]/pixelsPerStep[1])-1)
    print("numberOfSteps:", numberOfSteps)
    # Initialize a list to append window positions to
    x1 = lambda x: x_start_stop[0]+x*pixelsPerStep[1]
    y1 = lambda y: y_start_stop[0]+y*pixelsPerStep[0]
    x2 = lambda x: x1(x)+xy_window[1]
    y2 = lambda y: y1(y)+xy_window[0]
    window_list = [((x1(x),y1(y)),(x2(x),y2(y))) for x in range(0, numberOfSteps[1]) for y in range(0, numberOfSteps[0])]
    #print("window_list:", window_list)
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    return window_list

# Check the prediction time for a single sample
t=time.time()

#image = mpimg.imread('bbox-example-image.jpg')
imageNames=glob.glob("./test_images/*.jpg")
#imageNames=["./test_images/test1.jpg"]
#fig = plt.figure(figsize=(len(imageNames)*5, 2*5))
fig = plt.figure(figsize=(7,3))
for imageName,imageId in zip(imageNames, range(0, len(imageNames))):

    image = mpimg.imread(imageName)
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255

    MINIMUMWINDOW=64
    MAXIMUMWINDOW=128

    window_img=None
    windowSize=MINIMUMWINDOW
    hotWindowList=[]
    while windowSize <= MAXIMUMWINDOW:
        print ("windowSize:", windowSize)
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                            xy_window=(windowSize, windowSize), xy_overlap=(0.5, 0.5))

        hot_windows = search_windows(image, windows, classifier, scaler, color_space=FeatureVectorConfig.COLORSPACE, 
                                spatial_size=FeatureVectorConfig.SPATIALSIZE, hist_bins=FeatureVectorConfig.HISTOGRAMBINS, 
                                orient=FeatureVectorConfig.ORIENTATIONBINS, pix_per_cell=FeatureVectorConfig.PIXELSPERCELL, 
                                cell_per_block=FeatureVectorConfig.CELLSPERBLOCK, 
                                hog_channel=FeatureVectorConfig.HOGCHANNEL, spatial_feat=FeatureVectorConfig.SPATIALFEATURES, 
                                hist_feat=FeatureVectorConfig.HISTOGRAMFEATURES, hog_feat=FeatureVectorConfig.HOGFEATURES)                       

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        windowSize = windowSize*2
        hotWindowList+=hot_windows
        
    #plt.imshow(window_img)
    #plt.show()

    print("hotWindowList:", hotWindowList)
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            # heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1 # all channels
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0], 0] += 1 # red channel only

        # Return updated heatmap
        #print("heatmap shape:",heatmap.shape, ", type:", heatmap.dtype)
        #print("heatmap-counts:", np.unique(heatmap[:,:,0], return_counts=True), np.unique(heatmap[:,:,1], return_counts=True), np.unique(heatmap[:,:,2], return_counts=True))
        maxCount=np.max(heatmap[:,:,0])
        heatmap[:,:,0] = np.uint8(255.*heatmap[:,:,0]/maxCount)
        #print("heatmap-counts:", np.unique(heatmap[:,:,0], return_counts=True), np.unique(heatmap[:,:,1], return_counts=True), np.unique(heatmap[:,:,2], return_counts=True), ", maxCount:", maxCount)
        return heatmap

    heatmap=np.zeros_like(window_img)
    heatmap=add_heat(heatmap,hotWindowList)

    #plt.subplot(len(imageNames),2,imageId*2+1)
    plt.subplot(1,2,1)
    plt.imshow(window_img)
    plt.title(imageName)
    #plt.subplot(len(imageNames),2,imageId*2+2)
    plt.subplot(1,2,2)
    plt.imshow(heatmap, cmap='hot')
    plt.title(imageName)
    #fig.tight_layout()
    plt.imshow(heatmap)

    plt.show()
    imageNamePath=imageName.split('/')
    print("imageNamePath:", imageNamePath, ", imageNamePath[-1]:", imageNamePath[-1])
    fig.savefig("./output_images/slideWindows/"+imageNamePath[-1])
    #plt.savefig("./output_images/WindowSearch.jpg")
