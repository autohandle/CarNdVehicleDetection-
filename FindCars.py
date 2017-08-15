import numpy as np
import cv2
import HogFeatures
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

def combineImages(img, initial_img, α=0.8, λ=0.):
    return cv2.addWeighted(initial_img, α, img, 1.-α, λ)

def convert_color(image, conv):
    convertedImage=cv2.cvtColor(image, eval("cv2.COLOR_"+conv))
    return convertedImage

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    cv2.rectangle(draw_img,(0, ystart),(draw_img.shape[1],ystop),(0,255,0),6) 
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    print("find_cars-ystart:", ystart, ", ystop:", ystop, ", scale:", scale,
        ", ctrans_tosearch.shape:", ctrans_tosearch.shape, ", img_tosearch.shape:", img_tosearch.shape)
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

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
    hog2 = HogFeatures.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = HogFeatures.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    boundingBoxList=[]
    
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

            xleft = xpos*pix_per_cell # xleft,ytop in pixels
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = [] # bin_spatial(subimg, size=spatial_size)
            hist_features = [] # color_hist(subimg, nbins=hist_bins)

            #if X_scaler==None:
            #    X = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1).astype(np.float64)
            #    X = np.concatenate([spatial_features, hist_features, hog_features]) # features
            #    #print("find_cars-build scaler-X.shape):", X.shape)                        
            #    # Fit a per-column scaler
            #    scaler = StandardScaler().fit(X.reshape(-1, 1))
            #    #print('scaler: ', scaler, ", get_params:", scaler.get_params(deep=True), ", mean:", scaler.mean_, ", scale:", scaler.scale_)
            #else:
            #    scaler=X_scaler

            # Scale features and make a prediction
            #test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                #print("find_cars-xb:", xb, "steps , xpos:", xpos, "cells, yb:", yb, "steps , ypos:", ypos, "cells, xleft:", xleft
                #    , "pixels, ytop:", ytop, "pixels, window:", window,"pixels")
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #print("find_cars-draw at:", (xbox_left, ytop_draw+ystart), ",", (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                bounding_box=[(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)]
                boundingBoxList+=[bounding_box]
                cv2.rectangle(draw_img,bounding_box[0], bounding_box[1],(0,0,255),6)
    if TESTING : print("find_cars-boundingBoxList:",boundingBoxList)
    return boundingBoxList

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

def applyThreshold(heatmap, threshold):
    # Zero out pixels below the threshold
    print("applyThreshold-threshold:", threshold)
    thresholdMap=heatmap.copy()
    thresholdMap[thresholdMap <= threshold] = 0
    # Return thresholded map
    print("applyThreshold-threshold:", threshold, ", heatmap-counts:", np.unique(thresholdMap, return_counts=True))
    return thresholdMap

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
    labels=findLabels(heatMap)
    boundingBoxes=findBoundingBoxesFromLabels(labels)
    return drawBoxes(imageToLabel, boundingBoxes, color, thick), boundingBoxes  # drawBoxes makes an image copy
    #return imageToLabel # even though it was written on already

def makeThresholdMap(image, findCars, scales=[1.5], percentOfHeapmapToToss=.5):
    print("scales:", scales, ", type:", type(scales), "image.shape:", image.shape, ", dtype:", image.dtype, ", percentOfHeapmapToToss:", percentOfHeapmapToToss)
    boundingBoxList=[]
    for scale in scales:
        boundingBoxList += findCars(image, scale)
    heatMap=addHeat(image.shape, boundingBoxList)
    heatMapCounts=np.unique(heatMap, return_counts=True)
    print("makeThresholdMap-heatMapCounts:", heatMapCounts, ", len(heatMapCounts):", len(heatMapCounts), ", len(heatMapCounts[0]):", len(heatMapCounts[0]))
    heatMapMidpoint=heatMapCounts[0][int(round(len(heatMapCounts[0])*percentOfHeapmapToToss))]
    thresholdMap=applyThreshold(heatMap, heatMapMidpoint)
    print("makeThresholdMap-thresholdMap counts:", (np.unique(thresholdMap, return_counts=True)), ", len(thresholdMap):", len(thresholdMap), ", len(thresholdMap[0]):", len(thresholdMap[0]))
    return thresholdMap, boundingBoxList, heatMap

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



