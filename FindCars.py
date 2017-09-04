import numpy as np
import cv2
import HogFeatures
import FeatureVectorConfig
from scipy.ndimage.measurements import label

TESTING=False

#https://stackoverflow.com/questions/21275714/check-rectangle-inside-rectangle-in-python
#def contains(r1, r2):
#   return r1.x1 < r2.x1 < r2.x2 < r1.x2 and r1.y1 < r2.y1 < r2.y2 < r1.y2
def isCompletelyContainedBy(r1, r2): # r1 contains r2?
    r1x1=r1[0][0]
    r1y1=r1[0][1]
    r1x2=r1[1][0]
    r1y2=r1[1][1]

    r2x1=r2[0][0]
    r2y1=r2[0][1]
    r2x2=r2[1][0]
    r2y2=r2[1][1]

    return r1x1 < r2x1 < r2x2 < r1x2 and r1y1 < r2y1 < r2y2 < r1y2

def isEitherCompletelyContainedBy(r1, r2): # r1 contains r2 or r2 contains r1
    return isCompletelyContainedBy(r1,r2) or isCompletelyContainedBy(r2,r1)

# http://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/
def union(a,b):
    #print("union-a:", a, ", b:", b, ", a[0]:", a[0], ", b[0];", b[0])
    x1 = min(a[0][0], b[0][0])
    y1 = min(a[0][1], b[0][1])
    x2 = max(a[1][0], b[1][0])
    y2 = max(a[1][1], b[1][1])
    return ((x1, y1), (x2, y2))

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)

# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def computeIntersectionOverUnion(boundingBox1, boundingBox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])

    bb1=dict([('x1', boundingBox1[0][0]), ('x2', boundingBox1[1][0]), ('y1', boundingBox1[0][1]), ('y2', boundingBox1[1][1])])
    bb2=dict([('x1', boundingBox2[0][0]), ('x2', boundingBox2[1][0]), ('y1', boundingBox2[0][1]), ('y2', boundingBox2[1][1])])
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

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

def combineImages(img, initial_img, α=0.8, λ=0.):
    return cv2.addWeighted(initial_img, α, img, 1.-α, λ)

def convert_color(image, color_space):
    #print("convert_color-max(image):",np.max(image),", min(image):", np.min(image), ", color_space:", color_space)
    #print ("convert_color-conv:", conv, "eval(cv2.COLOR_+conv):", eval("cv2.COLOR_"+conv))  
    if False  :
        convertedImage=cv2.cvtColor(image, eval("cv2.COLOR_RGB2"+color_space))
        #print ("convert_color-convertedImage:", convertedImage)    
    else :
        if color_space != 'RGB':
            if color_space == 'HSV':
                convertedImage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                convertedImage = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                convertedImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                convertedImage = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                convertedImage = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: convertedImage = np.copy(image)
    #for channel in range(0,convertedImage.shape[2]):
    #    print("convert_color-max(convertedImage["+str(channel)+"]):",np.max(convertedImage[channel]),", min(convertedImage):", np.min(convertedImage[channel]), ", convertedImage.shape:", convertedImage.shape)
    return convertedImage

from sklearn.preprocessing import StandardScaler

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, decisionFunctionThreshold=0.0):
    
    draw_img = np.copy(img)
    #print("find_cars-max(img):",np.max(img),", min(img):", np.min(img))
    # green bounding box for search
    cv2.rectangle(draw_img,(0, ystart),(draw_img.shape[1],ystop),(0,255,0), 6) 
    #img = img.astype(np.float32)/255
    img = img.astype(np.float32)
    img_tosearch = img[ystart:ystop,:,:]
    #print ("img_tosearch:", img_tosearch)    
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = convert_color(img_tosearch, color_space=FeatureVectorConfig.COLORSPACE)
    #ctrans_tosearch = convert_color(img_tosearch, color_space="RGB")
    #ctrans_tosearch=img # the color conversion to HLS is making a negative S channel, so ignoring color conversion
    #print ("ctrans_tosearch:", ctrans_tosearch)    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    print("find_cars-ystart:", ystart, ", ystop:", ystop, ", scale:", scale,
        ", ctrans_tosearch.shape:", ctrans_tosearch.shape, ", img_tosearch.shape:", img_tosearch.shape)
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    #print("find_cars-max(ch2):",np.max(ch2),", min(ch2):", np.min(ch2))
    ch3 = ctrans_tosearch[:,:,2]
    #print("find_cars-max(ch3):",np.max(ch3),", min(ch3):", np.min(ch3))

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = HogFeatures.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #print("find_cars-max(hog1):",np.max(hog1),", min(hog1):", np.min(hog1))
    hog2 = HogFeatures.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #print("find_cars-max(hog2):",np.max(hog2),", min(hog2):", np.min(hog2))
    hog3 = HogFeatures.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #print("find_cars-max(hog3):",np.max(hog3),", min(hog3):", np.min(hog3))

    boundingBoxList=[]
    decisionFunctionList=[]
    
    if TESTING : print("find_cars-nxsteps:",nxsteps, ", nysteps:", nysteps, ", nxsteps*nysteps:", nxsteps*nysteps)
    for xb in range(nxsteps): # xb/yb in steps
        #print("find_cars-xb:", xb, ", xpos = xb*cells_per_step:", (xb*cells_per_step), ", xleft = xpos*pix_per_cell:", ((xb*cells_per_step)*pix_per_cell))
        for yb in range(nysteps):
            ypos = yb*cells_per_step # xpos/ypos in cells
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            #print ("hog_features:", hog_features)

            xleft = xpos*pix_per_cell # xleft,ytop in pixels
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = [] # bin_spatial(subimg, size=spatial_size)
            if FeatureVectorConfig.SPATIALFEATURES: spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = [] # color_hist(subimg, nbins=hist_bins)
            if FeatureVectorConfig.HISTOGRAMFEATURES: hist_features = color_hist(subimg, nbins=hist_bins)

            scaler=None
            if X_scaler==None:
                X = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1).astype(np.float64)
                X = np.concatenate([spatial_features, hist_features, hog_features]) # features
                print("find_cars-build scaler-X.shape):", X.shape, "X:", X)                        
                # Fit a per-column scaler
                scaler = StandardScaler().fit(X.reshape(-1, 1))
                #print('scaler: ', scaler, ", get_params:", scaler.get_params(deep=True), ", mean:", scaler.mean_, ", scale:", scaler.scale_)
            else:
                scaler=X_scaler
    
            # Scale features and make a prediction
            #print("find_cars-max(hog_features):",np.max(hog_features),", min(hog_features):", np.min(hog_features))
            test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            #print("find_cars-max(test_features):",np.max(test_features),", min(test_features):", np.min(test_features), ", test_features.shape:", test_features.shape)
            #print("find_cars-len(spatial_features):",len(spatial_features),", len(hist_features):", len(hist_features), ", len(hog_features):", len(hog_features))
            test_features = scaler.transform(test_features)    
            #print ("xb:", xb, ", yb:", yb, ", len(test_features):", len(hog_features))
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                decision_function = svc.decision_function(test_features)
                if decision_function >= decisionFunctionThreshold:
                    decisionFunctionList+=[decision_function]
                    #print("find_cars-xb:", xb, "steps , xpos:", xpos, "cells, yb:", yb, "steps , ypos:", ypos, "cells, xleft:", xleft
                    #    , "pixels, ytop:", ytop, "pixels, window:", window,"pixels")
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    #print("find_cars-draw at:", (xbox_left, ytop_draw+ystart), ",", (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                    bounding_box=[(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)]
                    print("find_cars-test_prediction:", test_prediction, ", decision_function:", decision_function, ", bounding_box:", bounding_box)
                    boundingBoxList+=[bounding_box]
                    cv2.rectangle(draw_img,bounding_box[0], bounding_box[1],(0,0,255),6)
                    #if TESTING : print("find_cars-len(boundingBoxList):",len(boundingBoxList), ", len(decisionFunctionList):", len(decisionFunctionList))

    if TESTING : print("find_cars-boundingBoxList:",boundingBoxList, ", decisionFunctionList:", decisionFunctionList)
    print("find_cars-len(boundingBoxList):", len(boundingBoxList), ", len(decisionFunctionList):", len(decisionFunctionList))
    assert len(boundingBoxList)==len(decisionFunctionList)
    return boundingBoxList, decisionFunctionList

# Here is your draw_boxes function from the previous exercise
def drawBoxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    print("drawBoxes-bboxes:", bboxes)
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def addHeat(imageShape, bbox_list):
    print("addHeat-imageShape:", imageShape)
    heatmap=np.zeros((imageShape[0], imageShape[1]), dtype=int)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1 # all channels
        # heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0], 0] += 1 # red channel only for 3 channel map

    # Return updated heatmap
    #print("heatmap shape:",heatmap.shape, ", type:", heatmap.dtype)
    #print("heatmap-counts:", np.unique(heatmap[:,:,0], return_counts=True), np.unique(heatmap[:,:,1], return_counts=True), np.unique(heatmap[:,:,2], return_counts=True))
    #maxCount=np.max(heatmap[:,:,0])
    #heatmap[:,:,0] = np.uint8(255.*heatmap[:,:,0]/maxCount) # for 3 channel map

    #heatmap = np.uint8(255.*heatmap/np.max(heatmap))

    #print("addHeat-heatmap-counts:", np.unique(heatmap[:,:,0], return_counts=True), np.unique(heatmap[:,:,1], return_counts=True), np.unique(heatmap[:,:,2], return_counts=True), ", maxCount:", maxCount)
    print("addHeat-heatmap-counts:", np.unique(heatmap, return_counts=True))
    return heatmap

    heatmap=np.zeros_like(window_img)
    heatmap=add_heat(heatmap,hotWindowList)

def addWeightedHeat(imageShape, bbox_list, bbox_weights):
    print("addWeightedHeat-len(bbox_list):", len(bbox_list), ", len(bbox_weights):", len(bbox_weights))
    assert len(bbox_list)==len(bbox_weights)
    print("addWeightedHeat-imageShape:", imageShape)
    heatmap=np.zeros((imageShape[0], imageShape[1]), dtype=int)
    if TESTING: print("addWeightedHeat-create-heatmap-counts:", np.unique(heatmap, return_counts=True))
    # Iterate through list of bboxes
    for box,weight in zip(bbox_list,bbox_weights) :
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        weightedHeat=int(round(weight[0]*10))
        if TESTING: print("addWeightedHeat-increment-box:", box, ", weight:", weight, ", weightedHeat:", weightedHeat)
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weightedHeat # all channels
        # heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0], 0] += 1 # red channel only for 3 channel map

    if TESTING: print("addWeightedHeat-heatmap-final-counts:", np.unique(heatmap, return_counts=True))
    return heatmap

HEATMAPSTACKSIZE=8
heatMapStack=None
heatMapStackOpenSlot=0 # next open slot

def clearHeatmapStack():
    global heatMapStack
    heatMapStack=None

def totalHeatmapStack(newHeatMap):
    global heatMapStack
    global heatMapStackOpenSlot
    global HEATMAPSTACKSIZE

    if heatMapStack is None:
        #heatMapStack=np.ndarray(shape=(HEATMAPSTACKSIZE,newHeatMap.shape[0], newHeatMap.shape[1]))
        heatMapStack=np.empty(shape=(HEATMAPSTACKSIZE,newHeatMap.shape[0], newHeatMap.shape[1]))
        print ("totalHeatmapStack-newHeatMap.shape:", newHeatMap.shape, ", heatMapStack.shape:", heatMapStack.shape)

    # inset new heatmap into stack
    heatMapStack[heatMapStackOpenSlot]=newHeatMap
    heatMapStackOpenSlot=(heatMapStackOpenSlot+1)%HEATMAPSTACKSIZE

    totalMap=None
    mapCount=0
    for map in heatMapStack:
        if totalMap is None:
            totalMap=map
            mapCount+=1
        else:
            if map is not None:
                totalMap+=map
                mapCount+=1

    if TESTING: print("totalHeatmapStack-mapCount:", mapCount, ", totalMap counts:", (np.unique(totalMap, return_counts=True)), ", len(totalMap):", len(totalMap), ", len(totalMap[0]):", len(totalMap[0]))
    print("totalHeatmapStack-mapCount:", mapCount)
    return totalMap, mapCount # mapCount is incorrect until stack fills

def applyThreshold(heatmap, threshold):
    # Zero out pixels below the threshold
    print("applyThreshold-threshold:", threshold)
    thresholdMap=heatmap.copy()
    thresholdMap[thresholdMap <= threshold] = 0
    # Return thresholded map
    if TESTING: print("applyThreshold-threshold:", threshold, ", heatmap-counts:", np.unique(thresholdMap, return_counts=True))
    return thresholdMap

def normalizeMap(map):
    normalizedMap = np.uint8(255*map/np.max(map))
    return normalizedMap

def findBoundingBoxesFromLabels(labels):
    # Iterate through all detected cars
    boundingBoxList=[]
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boundingBoxList+=[bbox]
        # Draw the box on the image
        #cv2.rectangle(img, bbox[0], bbox[1], lineColor, 6)
    # Return the image
    return boundingBoxList

def findLabels(heatMap):
        # Visualize the heatmap when displaying    
    thresholdMap = np.clip(heatMap, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(thresholdMap) # labels[0] is the bit map, labelss[1] is the number of maps
    print("findLabels-len(labels):", len(labels), ", labels[1]:", labels[1])
    return labels

def drawLabelsOnImage(imageToLabel, heatMap, color=(0,255,0), thick=6):
    labels=findLabels(heatMap) # labels[0] is the bit map, labelss[1] is the number of maps
    boundingBoxes=findBoundingBoxesFromLabels(labels)
    return drawBoxes(imageToLabel, boundingBoxes, color, thick), boundingBoxes  # drawBoxes makes an image copy
    #return imageToLabel # even though it was written on already

USEBOUNDINGBOXWEIGHTS=True
USESTACKOFHEATMAPS=True
def makeThresholdMap(image, findCars, scales=[1.5], percentOfHeapmapToToss=.5):
    print("scales:", scales, ", type:", type(scales), "image.shape:", image.shape, ", dtype:", image.dtype, ", percentOfHeapmapToToss:", percentOfHeapmapToToss)
    boundingBoxList=[]
    boundingBoxWeights=[]
    for scale in scales:
        listOfBoundingBoxes, listOfWeights = findCars(image, scale)
        boundingBoxList+=listOfBoundingBoxes
        boundingBoxWeights+=listOfWeights

    if USEBOUNDINGBOXWEIGHTS:
        unNormalizedHeatMap=addWeightedHeat(image.shape, boundingBoxList, boundingBoxWeights)
    else:
        unNormalizedHeatMap=addHeat(image.shape, boundingBoxList)

    if USESTACKOFHEATMAPS:
        unNormalizedHeatMap,_=totalHeatmapStack(unNormalizedHeatMap)


    unNormalizedHeatMapCounts=np.unique(unNormalizedHeatMap, return_counts=True)
    if TESTING: print("makeThresholdMap-unNormalizedHeatMapCounts:", unNormalizedHeatMapCounts, ", len(unNormalizedHeatMapCounts):", len(unNormalizedHeatMapCounts), ", len(unNormalizedHeatMapCounts[0]):", len(unNormalizedHeatMapCounts[0]))
    unNormalizedHeatMapMidpoint=unNormalizedHeatMapCounts[0][int(round(len(unNormalizedHeatMapCounts[0])*percentOfHeapmapToToss))]
    thresholdMap=applyThreshold(unNormalizedHeatMap, unNormalizedHeatMapMidpoint)
    print("makeThresholdMap-max(thresholdMap):", np.max(thresholdMap), ", min(thresholdMap):", np.min(thresholdMap))
    if TESTING: print("makeThresholdMap-thresholdMap counts:", (np.unique(thresholdMap, return_counts=True)), ", len(thresholdMap):", len(thresholdMap), ", len(thresholdMap[0]):", len(thresholdMap[0]))
    normalizedMap=normalizeMap(thresholdMap)
    if TESTING: print("makeThresholdMap-normalizedMap counts:", (np.unique(normalizedMap, return_counts=True)), ", len(normalizedMap):", len(normalizedMap), ", len(normalizedMap[0]):", len(normalizedMap[0]))
    print("makeThresholdMap-max(normalizedMap):", np.max(normalizedMap), ", min(normalizedMap):", np.min(normalizedMap))
    return normalizedMap, boundingBoxList, unNormalizedHeatMap, boundingBoxWeights

def testBoundingBox():
    bb1=((0,0), (1,1))
    bb2=((0,0), (1,1))
    bb3=((0,.5), (1,1))
    bb4=((1,1), (2,2))
    bb1xbb2=computeIntersectionOverUnion(bb1, bb2)
    print("bb1xbb2:", bb1xbb2)
    assert bb1xbb2==1.0
    bb1xbb3=computeIntersectionOverUnion(bb1, bb3)
    print("bb1xbb3:", bb1xbb3)
    assert bb1xbb3==0.5
    bb1xbb4=computeIntersectionOverUnion(bb1, bb4)
    print("bb1xbb4:", bb1xbb4)
    assert bb1xbb4==0.

    bb1nbb2=union(bb1, bb2)
    print("bb1:", bb1)
    print("bb1nbb2:", bb1nbb2)
    np.testing.assert_equal(bb1,bb1nbb2)
    bb1nbb3=union(bb1, bb3)
    print("bb1xbb3:", bb1xbb3)
    np.testing.assert_equal(bb1,bb1nbb3)
    bb1nbb4=union(bb1, bb4)
    print("bb1nbb4:", bb1nbb4)
    bb1pbb4=((bb1[0]), (bb4[1]))
    print("bb1pbb4:", bb1pbb4)
    np.testing.assert_equal(bb1pbb4, bb1nbb4)

#testBoundingBox()



