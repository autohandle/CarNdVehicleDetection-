import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import numpy as np
#import pickle
import cv2
#from lesson_functions import *
import FeatureVectorConfig
#import HogFeatures
import SaveAndRestoreClassifier

#from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
import FindCars

LINEAR=True

if LINEAR:
    MODELFILENAME='./models/Linear.svm'
    SCALERFILENAME='./models/LinearScaler.svm'
    SCALES=[1.5] # Linear
    SCALES=np.linspace(.75, 2., 6) # too many boounding boxes, esp at .75
    SCALES=np.linspace(1., 2., 5) # too many boounding boxes, esp at .75
    PERCENTOFBOUNDINGBOXESTOTOSS=.1 # Linear
    PERCENTOFBOUNDINGBOXESTOTOSS=len(SCALES)/10.

else:
    MODELFILENAME='./models/RbfClassifier.svm'
    SCALERFILENAME='./models/RbfScaler.svm'
    PERCENTOFBOUNDINGBOXESTOTOSS=.3 # Rbf
    SCALES=[2.5] # Rbf

from sklearn.externals import joblib

print('restoring model from: ', MODELFILENAME)
svc= SaveAndRestoreClassifier.restoreClassifier(MODELFILENAME)


print('restoring scaler from: ', SCALERFILENAME)
X_scaler= SaveAndRestoreClassifier.restoreScaler(SCALERFILENAME)
print('X_scaler: ', X_scaler, ", get_params:", X_scaler.get_params(deep=True), ", mean:", X_scaler.mean_, ", scale:", X_scaler.scale_)

#dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
#svc = dist_pickle["svc"]
#X_scaler = dist_pickle["scaler"]
#orient = dist_pickle["orient"]
orient=FeatureVectorConfig.ORIENTATIONBINS
#pix_per_cell = dist_pickle["pix_per_cell"]
pix_per_cell = FeatureVectorConfig.PIXELSPERCELL
#cell_per_block = dist_pickle["cell_per_block"]
cell_per_block = FeatureVectorConfig.CELLSPERBLOCK
#spatial_size = dist_pickle["spatial_size"]
spatial_size = FeatureVectorConfig.SPATIALSIZE
#hist_bins = dist_pickle["hist_bins"]
hist_bins = FeatureVectorConfig.HISTOGRAMBINS

ystart = 400
ystop = 656

    
#out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

boundingBoxList=[]
#SCALES=[1., 1.5, 1.75]
#SCALES=np.linspace(.75, 2., 6)
#img = mpimg.imread('test_image.jpg')
imageNames=glob.glob("./test_images/*.jpg")
#imageNames=["./test_images/test1.jpg"]
fig = plt.figure(figsize=(len(imageNames)*5, 2*5))
for imageName,imageId in zip(imageNames, range(0, len(imageNames))):
    img = mpimg.imread(imageName)

    #def makeThresholdMap(image, findCars, scales=[1.5], percentOfHeapmapToToss=.5):
    findCars=lambda image, scale: FindCars.find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    thresholdMap,_,heatMap =FindCars.makeThresholdMap(img, findCars, SCALES, PERCENTOFBOUNDINGBOXESTOTOSS)

    #for scale in SCALES:
    #    boundingBoxList += FindCars.find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    #out_img=FindCars.drawBoxes(img, boundingBoxList) # blue boxes
    #out_img=img # skip the blue boxes
    #heatMap=FindCars.addHeat(img.shape, boundingBoxList)
    #heatMapCounts=np.unique(heatMap, return_counts=True)
    #print("heatMapCounts:", heatMapCounts, ", len(heatMapCounts):", len(heatMapCounts), ", len(heatMapCounts[0]):", len(heatMapCounts[0]))
    #heatMapMidpoint=heatMapCounts[0][len(heatMapCounts[0])//4]
    #thresholdMap=FindCars.applyThreshold(heatMap, heatMapMidpoint)

    # Visualize the heatmap when displaying    
    #thresholdMap = np.clip(thresholdMap, 0, 255)
    # Find final boxes from heatmap using label function
    #labels = label(thresholdMap)
    #print("len(labels):", len(labels), ", labels[1]:", labels[1])

    #out_img=FindCars.drawLabelsOnImage(out_img, thresholdMap) # drawLabelsOnImage makes a copy of the image
    out_img=FindCars.drawLabelsOnImage(img, thresholdMap) # drawLabelsOnImage makes a copy of the image
    cv2.rectangle(out_img,(0, ystart),(out_img.shape[1],ystop),(0,255,255),6) 


    figureColumnCount=3
    figureRowCount=1
    showImages = plt.figure(figsize = (figureColumnCount*5,figureRowCount*5))

    imageIndex=0

    p=showImages.add_subplot(figureRowCount,figureColumnCount,imageIndex+1)
    p.set_title(imageName+"\nthreshold image")
    p.imshow(out_img)

    p=showImages.add_subplot(figureRowCount,figureColumnCount,imageIndex+2)
    p.set_title(imageName+"\nheatmap image")
    p.imshow(heatMap, cmap='hot')

    p=showImages.add_subplot(figureRowCount,figureColumnCount,imageIndex+3)
    p.set_title(imageName+"\nthreshold heatmap")
    p.imshow(thresholdMap, cmap='hot')

    plt.tight_layout()
    plt.show()

plt.close()
