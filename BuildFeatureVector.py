import matplotlib.image as mpimg
import cv2
import numpy as np
import re

import HogFeatures

TESTING=False

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    #print("len(hist_features):", len(hist_features))
    return hist_features

def printSingleImageFeatures(feature_image, color_space):
    print("singleImageFeatures-color_space:", color_space)
    for channel in range(0,feature_image.shape[2]): \
        print("singleImageFeatures-max(feature_image["+str(channel)+"]):",np.max(feature_image[channel]),", min(feature_image):", np.min(feature_image[channel]), ", feature_image.shape:", feature_image.shape)


def singleImageFeatures(img, color_space='RGB', spatial_size=(32, 32),
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
    if TESTING : printSingleImageFeatures(feature_image, color_space)

    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
        #print("singleImageFeatures-len(spatial_features):", len(spatial_features), ", spatial_size:", spatial_size)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
        #print("singleImageFeatures-len(hist_features):", len(hist_features), ", hist_bins:", hist_bins)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(HogFeatures.get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = HogFeatures.get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #print("singleImageFeatures-len(hog_features):", len(hog_features))
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def rescalePng(pngImage):
    return np.uint8(255.*pngImage/np.max(pngImage))



def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
    hist_bins=32, orient=9,
    pix_per_cell=8, cell_per_block=2, hog_channel=0,
    spatial_feat=False, hist_feat=False, hog_feat=True):

    # Create a list to append feature vectors to
    features = []
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        pngMatch = re.search('.*\\.png', file)
        #print("extract_features-pngMatch?", pngMatch,", file:", file)
        if pngMatch:
            image=rescalePng(image)
            #print("extract_features-pngImage-counts:", np.unique(image[:,:,0], return_counts=True), np.unique(image[:,:,1], return_counts=True), np.unique(image[:,:,2], return_counts=True))
        imageFeatures=singleImageFeatures(image, color_space, spatial_size,
                        hist_bins, orient,
                        pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat)
        features.append(imageFeatures)
    print ("extract_features-len(features):", len(features),", len(features[0]):", len(features[0]),
        ", spatial_feat:", spatial_feat, ", hist_feat:", hist_feat, ", hog_feat:", hog_feat)
    return features
