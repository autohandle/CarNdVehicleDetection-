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
    #SCALES=np.linspace(.75, 2., 6) # too many boounding boxes, esp at .75, .75 doesn't seem to add anything
    SCALES=np.linspace(1., 2., 5)
    PERCENTOFBOUNDINGBOXESTOTOSS=len(SCALES)/10.
    PERCENTOFBOUNDINGBOXESTOTOSS=.5 # Linear

else:
    MODELFILENAME='./models/RbfClassifier.svm'
    SCALERFILENAME='./models/RbfScaler.svm'
    PERCENTOFBOUNDINGBOXESTOTOSS=.5 # Rbf
    SCALES=np.linspace(1., 2., 5)

from sklearn.externals import joblib

print('restoring model from: ', MODELFILENAME)
svc= SaveAndRestoreClassifier.restoreClassifier(MODELFILENAME)


print('restoring scaler from: ', SCALERFILENAME)
X_scaler= SaveAndRestoreClassifier.restoreScaler(SCALERFILENAME)
print('X_scaler: ', X_scaler, ", get_params:", X_scaler.get_params(deep=True),
    ", mean:", X_scaler.mean_, ", len(mean):", len(X_scaler.mean_),
    ", scale:", X_scaler.scale_, ", len(scale):", len(X_scaler.scale_))

#dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
#svc = dist_pickle["svc"]
#X_scaler = dist_pickle["scaler"]
#orient = dist_pickle["orient"]
ORIENT=FeatureVectorConfig.ORIENTATIONBINS
#pix_per_cell = dist_pickle["pix_per_cell"]
PIX_PER_CELL = FeatureVectorConfig.PIXELSPERCELL
#cell_per_block = dist_pickle["cell_per_block"]
CELL_PER_BLOCK = FeatureVectorConfig.CELLSPERBLOCK
#spatial_size = dist_pickle["spatial_size"]
SPATIAL_SIZE = FeatureVectorConfig.SPATIALSIZE
#hist_bins = dist_pickle["hist_bins"]
HIST_BINS = FeatureVectorConfig.HISTOGRAMBINS
SVCDECISIONFUNCTIONTHRESHOLD=.8

ystart = 400
ystop = 656

    
#out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS)

boundingBoxList=[]
#SCALES=[1., 1.5, 1.75]
#SCALES=np.linspace(.75, 2., 6)
#img = mpimg.imread('test_image.jpg')
imageNames=glob.glob("./test_images/*.jpg")
#imageNames=["./test_images/test1.jpg"]
#fig = plt.figure(figsize=(len(imageNames)*5, 2*5))
for imageName,imageId in zip(imageNames, range(0, len(imageNames))):
    FindCars.clearHeatmapStack()
    img = mpimg.imread(imageName)

    #def makeThresholdMap(image, findCars, scales=[1.5], percentOfHeapmapToToss=.5):
    findCars=lambda image, scale: FindCars.find_cars(image, ystart, ystop, scale, svc, X_scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, SVCDECISIONFUNCTIONTHRESHOLD)
    thresholdMap,boundingBoxList,heatMap,_ =FindCars.makeThresholdMap(img, findCars, SCALES, PERCENTOFBOUNDINGBOXESTOTOSS)

    #for scale in SCALES:
    #    boundingBoxList += FindCars.find_cars(img, ystart, ystop, scale, svc, X_scaler, ORIENT, pix_per_cell, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS)

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
    out_img, boundingBoxes=FindCars.drawLabelsOnImage(img, thresholdMap) # drawLabelsOnImage makes a copy of the image
    #print ("out_img-type:", type(out_img), ", len:", len(out_img), ", out_img:", out_img)
    cv2.rectangle(out_img,(0, ystart),(out_img.shape[1],ystop),(0,255,255),6) 
    # no good, because it is not weighted
    #out_img=FindCars.drawBoxes(out_img, boundingBoxList, color=(0, 0, 255), thick=6)


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

    imageNamePath=imageName.split('/')
    print("imageNamePath:", imageNamePath, ", imageNamePath[-1]:", imageNamePath[-1])
    showImages.savefig("./output_images/threshold/"+imageNamePath[-1])



#plt.close()
