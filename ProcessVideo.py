import matplotlib.image as mpimage

import numpy as np
import cv2

from moviepy.editor import VideoFileClip
import FeatureVectorConfig
import SaveAndRestoreClassifier
import FindCars

from sklearn.externals import joblib

LINEAR=True
TESTING=False

if LINEAR:
    MODELFILENAME='./models/Linear.svm'
    SCALERFILENAME='./models/LinearScaler.svm'
    SCALES=[1.5] # Linear
    #SCALES=np.linspace(.75, 2., 6)
    #SCALES=np.linspace(1., 2., 5)
    SCALES=np.linspace(1.2, 1.8, 4)
    PERCENTOFBOUNDINGBOXESTOTOSS=len(SCALES)*.125 # Linear


else:
    MODELFILENAME='./models/RbfClassifier.svm'
    SCALERFILENAME='./models/RbfScaler.svm'
    PERCENTOFBOUNDINGBOXESTOTOSS=.5 # Rbf
    SCALES=[2.5] # Rbf
    SCALES=np.linspace(1.2, 1.8, 4)


print('restoring model from: ', MODELFILENAME)
svc= SaveAndRestoreClassifier.restoreClassifier(MODELFILENAME)

print('restoring scaler from: ', SCALERFILENAME)
X_scaler= SaveAndRestoreClassifier.restoreScaler(SCALERFILENAME)
print('X_scaler: ', X_scaler, ", get_params:", X_scaler.get_params(deep=True), ", mean:", X_scaler.mean_, ", scale:", X_scaler.scale_)


frameNumber=None
ystart = 400
ystop = 656

thresholdMap=None
heatMap=None

import matplotlib.gridspec as gridspec
import matplotlib
#ValueError: Unrecognized backend string "qtagg": valid strings are ['GTKCairo', 'WebAgg', 'GTK3Agg', 'TkAgg', 'gdk',
#'GTKAgg', 'agg', 'WX', 'Qt5Agg', 'GTK', 'cairo', 'nbAgg', 'MacOSX', 'pdf', 'template', 'ps', 'GTK3Cairo', 'Qt4Agg',
#'svg', 'pgf', 'WXAgg']
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

def plotFigure(videoFrame, heatMap, thresholdMap, lambdaAx3):
    # Plot figure with subplots of different sizes
    fig = plt.figure(figsize=(20,10))
    # set up subplot grid: 3x2
    gridspec.GridSpec(3,2)

    # large subplot — filledLaneVideoFrame 2x2@0,0 on a 3x2
    ax1=plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=2)
    #print("ax1:", ax1)
    plt.imshow(videoFrame)

    # small subplot 1: annotatedVisualizationImage 1x1@2,0 on a 3x2
    ax2=plt.subplot2grid((3,2), (2,0))
    ax2.set_title("heat map")
    #print("ax2:", ax2)

    plt.imshow(heatMap, cmap='hot')

    # small subplot 2: annotatedVisualizationImage 1x1@2,1 on a 3x2
    ax3=plt.subplot2grid((3,2), (2,1))
    ax3.set_title("threshold map")
    #print("ax3:", ax3)
    lambdaAx3(ax3)
    plt.imshow(thresholdMap, cmap='hot')

    fig.tight_layout()

    return fig

# convert figure to image
# https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def convertFigureToImage(figure):
    print("convertFigureToImage-figure type:", type(figure))
    figure.canvas.draw()
    print("convertFigureToImage-draw")
    # Now we can save it to a numpy array.
    data = np.fromstring(figure.canvas.tostring_rgb(blit=False), dtype=np.uint8, sep='')
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    return data

BOUNDINGBOXOVERLAP=.5

def findBoundingBoxesInList(theBoundingBoxesList, theBoundingBox):
    overlappingBoundingBoxes=[]
    for boundingBox in theBoundingBoxesList:
        if TESTING : print("findBoundingBoxesInList-theBoundingBox:", theBoundingBox, ", boundingBox:", boundingBox)
        if TESTING : print("findBoundingBoxesInList-computeIntersectionOverUnion?", FindCars.computeIntersectionOverUnion(theBoundingBox, boundingBox))
        if FindCars.computeIntersectionOverUnion(boundingBox, theBoundingBox) > BOUNDINGBOXOVERLAP: #
            overlappingBoundingBoxes.append(boundingBox)
    if TESTING and (len(overlappingBoundingBoxes) > 0) : print("findBoundingBoxesInList-overlappingBoundingBoxes:", overlappingBoundingBoxes)
    return overlappingBoundingBoxes

def findBoundingBoxesInFrames(theFramesBoundingBoxArray, theBoundingBox): 
    boundingBoxesInFrames=[]
    for listOfBoundingBoxes in theFramesBoundingBoxArray:
        overlappingBoundingBoxes=findBoundingBoxesInList(listOfBoundingBoxes, theBoundingBox)
        if TESTING and (len(overlappingBoundingBoxes) > 0) : \
            print("findBoundingBoxesInFrames-len(listOfBoundingBoxes):", len(listOfBoundingBoxes), ", theBoundingBox:", theBoundingBox)
        boundingBoxesInFrames.extend(overlappingBoundingBoxes)
    #print("findBoundingBoxesInFrames-boundingBoxesInFrames:", boundingBoxesInFrames)
    return boundingBoxesInFrames;

def findOverlappingBoundingBoxes(theFramesBoundingBoxArray, theBoundingBoxList):
    #print("findOverlappingBoundingBoxes-len(theFramesBoundingBoxArray):", len(theFramesBoundingBoxArray),
    #    ", theBoundingBoxList:", theBoundingBoxList)
    overlappingBoundingBoxes=[] # a list for each boundingBox in theBoundingBoxList
    for boundingBox in theBoundingBoxList:
        if TESTING : print("findOverlappingBoundingBoxes-boundingBox:", boundingBox)
        boundingBoxesInAllFrames=findBoundingBoxesInFrames(theFramesBoundingBoxArray, boundingBox)
        overlappingBoundingBoxes.append(boundingBoxesInAllFrames) # entries are one to one with theBoundingBoxList
    print("findOverlappingBoundingBoxes-overlappingBoundingBoxes:", overlappingBoundingBoxes, ", len(theBoundingBoxList):", len(theBoundingBoxList))
    return overlappingBoundingBoxes

def mergeOverlappingBoundingBoxes(theBoundingBoxList, theBoundingBox):
    #print("mergeOverlappingBoundingBoxes-theBoundingBoxList:", theBoundingBoxList,
    #    ", theBoundingBox:", theBoundingBox)
    mergedBoundingBoxes=[] # nont one to one with theOverlappingBoundingBoxList, len(0) entries are ignored
    if (len(theBoundingBoxList) != 0):
        mergedBoundingBox=theBoundingBox
        for boundingBox in theBoundingBoxList:
            #print("mergeOverlappingBoundingBoxes-boundingBox:", boundingBox, ", mergedBoundingBox:", mergedBoundingBox)
            mergedBoundingBox=FindCars.union(mergedBoundingBox, boundingBox)
            mergedBoundingBoxes.append(mergedBoundingBox)
    if TESTING : print("mergeOverlappingBoundingBoxes-mergedBoundingBoxes:", mergedBoundingBoxes, \
        ", theBoundingBox:", theBoundingBox, ", theBoundingBoxList:", theBoundingBoxList)
    return mergedBoundingBoxes;

BOUNDINDBOXCOLLECTIONSIZE=20
OVERLAPTHRESHOLD=5

def mergeBoundingBoxesAcrossFrames(theFramesBoundingBoxArray, theBoundingBoxList): # boundingBoxCollection from labels
    global frameNumber
    mergedBoundingBoxes=[]
    overlappingBoundingBoxLists=findOverlappingBoundingBoxes(theFramesBoundingBoxArray, theBoundingBoxList) # list of a: list of all boundingBoxes for boxes that overlap
    #print("mergeBoundingBoxesAcrossFrames-len(overlappingBoundingBoxLists):", len(overlappingBoundingBoxLists), ", len(theBoundingBoxList):", len(theBoundingBoxList))
    for overlappingBoundingBoxList, boundingBox in zip(overlappingBoundingBoxLists, theBoundingBoxList):
        print("mergeBoundingBoxesAcrossFrames-len(overlappingBoundingBoxList):", len(overlappingBoundingBoxList),
            ", >= OVERLAPTHRESHOLD?", (len(overlappingBoundingBoxList) >= OVERLAPTHRESHOLD),
            ",overlappingBoundingBoxList:", overlappingBoundingBoxList,", boundingBox:", boundingBox),
        print("mergeBoundingBoxesAcrossFrames-frameNumber<BOUNDINDBOXCOLLECTIONSIZE?", (frameNumber<BOUNDINDBOXCOLLECTIONSIZE),
            ", len(overlappingBoundingBoxList) == frameNumber?", (len(overlappingBoundingBoxList) == frameNumber)),
        if (len(overlappingBoundingBoxList) > 0): # any overlapping boxes?
            if (    (len(overlappingBoundingBoxList) >= OVERLAPTHRESHOLD)
            or (frameNumber<BOUNDINDBOXCOLLECTIONSIZE and (len(overlappingBoundingBoxList) >= 2)) ): # then there is a overlapping bounding box in every frame
                mergedBoundingBox=mergeOverlappingBoundingBoxes(overlappingBoundingBoxList, boundingBox)
                mergedBoundingBoxes.append(mergedBoundingBox)
        #print("mergeBoundingBoxesAcrossFrames-mergedBoundingBoxes:", mergedBoundingBoxes)
    return mergedBoundingBoxes

currentBoxSlot=0
labelBoundingBoxCollection=np.empty((BOUNDINDBOXCOLLECTIONSIZE,), dtype=list) # array for frames of: list of bounding boxes
labelBoundingBoxCollection.fill(()) # array on empty lists
print("ProcessVideo-currentBoxSlot:", currentBoxSlot, ", len(labelBoundingBoxCollection):", len(labelBoundingBoxCollection))

def processLabelBoundingBoxes(theLabelBoundingBoxArray):
    global currentBoxSlot
    global labelBoundingBoxCollection
    print("processLabelBoundingBoxes-theLabelBoundingBoxArray:", theLabelBoundingBoxArray)
    mergedBoundingBoxes=mergeBoundingBoxesAcrossFrames(labelBoundingBoxCollection, theLabelBoundingBoxArray)
    labelBoundingBoxCollection[currentBoxSlot]=theLabelBoundingBoxArray
    currentBoxSlot=(currentBoxSlot+1)%BOUNDINDBOXCOLLECTIONSIZE
    print("processLabelBoundingBoxes-mergedBoundingBoxes:", mergedBoundingBoxes, ", labelBoundingBoxCollection:", labelBoundingBoxCollection)
    return mergedBoundingBoxes

def bumpFrameNumber():
    global frameNumber

    if (frameNumber is None):
        frameNumber=0
    else:
        frameNumber+=1
    return frameNumber

def process_image(videoFrame):
    global ystart
    global ystop
    global SCALES
    global svc
    global X_scaler

    global thresholdMap
    global heatMap

    global currentBoxSlot
    global labelBoundingBoxCollection

    frameNumber=bumpFrameNumber()

    findCars=lambda image,scale: FindCars.find_cars(image, ystart, ystop, scale, svc, X_scaler,
        FeatureVectorConfig.ORIENTATIONBINS,
        FeatureVectorConfig.PIXELSPERCELL,
        FeatureVectorConfig.CELLSPERBLOCK,
        FeatureVectorConfig.SPATIALSIZE,
        FeatureVectorConfig.HISTOGRAMBINS
        )
    thresholdMap,_,heatMap =FindCars.makeThresholdMap(videoFrame, findCars, SCALES, PERCENTOFBOUNDINGBOXESTOTOSS)

    #print("process_image-after makeThresholdMap")
    annotatedImage, labelBoundingBoxes=FindCars.drawLabelsOnImage(videoFrame, thresholdMap) # drawLabelsOnImage makes a copy of the image

    #print("process_image-labelBoundingBoxCollection.shape:", labelBoundingBoxCollection.shape, ", len(labelBoundingBoxes):", len(labelBoundingBoxes))
    #consolidatedBoundingBoxes=processLabelBoundingBoxes(labelBoundingBoxes)
    #print("process_image-currentBoxSlot:", currentBoxSlot, ", consolidatedBoundingBoxes:", consolidatedBoundingBoxes)

    #if len(consolidatedBoundingBoxes) > 0 : 
    #    annotatedImage=FindCars.drawBoxes(annotatedImage, consolidatedBoundingBoxes[0], color=(255,255,0))

    #print("process_image-after drawLabelsOnImage")
    annotateThresholdImage = lambda axes : axes.text(0.1, 0.9, ("frame: "+str(frameNumber)
                                                      ),
                                              fontsize=16,
                                              horizontalalignment='left',
                                              verticalalignment='top',
                                              transform = axes.transAxes,
                                              color='w')
    figure=plotFigure(annotatedImage, heatMap, thresholdMap, annotateThresholdImage)
    cv2.rectangle(annotatedImage,(0, ystart),(annotatedImage.shape[1],ystop),(0,255,255), 2)

    #return convertFigureToImage(figure) # doesn't wprk on mac
    return annotatedImage


def processProjectVideo():
    CHALLENGEVIDEOOUTPUT = 'test_videos_output/ChallengeVideoOutput%04d.mp4'
    PROJECTVIDEOOUTPUT = './output_images/video/ProjectVideoOutput_%04d-%04d.mp4'
    CLIPLENGTH=10 # process 10 second clips
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
    video = VideoFileClip('./project_video.mp4')
    #video = VideoFileClip("./test_video.mp4")
    duration = video.duration
    #duration=31.4
    numberOfClips=int(round(duration/float(CLIPLENGTH)+0.5))
    print ("processProjectVideo-duration:", duration, ", CLIPLENGTH:", CLIPLENGTH, ", numberOfClips:", numberOfClips)
    for clipNumber in range(0, numberOfClips): # 
        clipStart=clipNumber*CLIPLENGTH
        clipStop=min((clipNumber+1)*CLIPLENGTH,duration)
        print ("processProjectVideo-clipNumber:", clipNumber, ", clipStart:", clipStart, ", clipStop:", clipStop)
        videoClip = video.subclip(clipStart,clipStop)
        annotatedClip = videoClip.fl_image(process_image)
        videoFileName=PROJECTVIDEOOUTPUT % (clipStart,clipStop)
        print ("processProjectVideo-videoFileName:", videoFileName)
        annotatedClip.write_videofile(videoFileName, audio=False)

def testProcessImage():
    global frameNumber
    frameNumber=0
    videoFrameName="./test_images/test1.jpg"
    videoFrame=mpimage.imread(videoFrameName)
    print("testProcessImage-videoFrameName: ",videoFrameName, ", videoFrame.shape:", videoFrame.shape, ", type:", videoFrame.dtype)
    annotatedImage=process_image(videoFrame)
    #print("process_image-after process_image")
    print("testProcessImage-thresholdMap.type:", type(thresholdMap), ", heatMap.type:", type(heatMap), ", annotatedImage.type:", type(annotatedImage))
    print("testProcessImage-thresholdMap.shape:", thresholdMap.shape, ", type:", thresholdMap.dtype)
    print("testProcessImage-heatMap.shape:", heatMap.shape, ", type:", heatMap.dtype)
    annotateThresholdImage = lambda axes : axes.text(0.1, 0.9, ("frame: "+str(frameNumber)
                                                      ),
                                              fontsize=16,
                                              horizontalalignment='left',
                                              verticalalignment='top',
                                              transform = axes.transAxes,
                                              color='w')
    figure=plotFigure(annotatedImage, heatMap, thresholdMap, annotateThresholdImage)
    plt.show()
    plt.close(figure)

def testMergeBoundingBoxesAcrossFrames():
    global frameNumber
    global TESTING
    TESTING=True

    frameNumber=0
    labelBoundingBoxCollection= [    list([((336, 400), (431, 495)), ((720, 400), (1151, 543)), ((168, 520), (263, 543))]),
                                     list([((336, 400), (431, 495)), ((720, 400), (1151, 543)), ((168, 520), (263, 543))]),
                                     list([((696, 400), (1031, 519)), ((264, 448), (335, 543)), ((1056, 448), (1151, 495)), ((72, 496), (119, 567))]),
                                     list([((672, 400), (935, 543)), ((1008, 400), (1031, 495)), ((1080, 400), (1103, 495)), ((120, 472), (143, 567)), ((624, 496), (647, 519))]),
                                     list([((720, 400), (1175, 543)), ((48, 472), (143, 567))]),
                                     list([((360, 400), (479, 495)), ((720, 400), (1175, 543))]),
                                     list([((720, 400), (1175, 567))]),
                                     list([((192, 400), (335, 543)), ((360, 400), (479, 495)), ((696, 400), (1175, 543))])
                                ]
    print("testMergeBoundingBoxesAcrossFrames-labelBoundingBoxCollection:",labelBoundingBoxCollection)
    print("------------------------------")
    theLabelBoundingBoxArray= [((216, 400), (335, 543)), ((384, 400), (479, 495)), ((840, 400), (911, 495)), ((936, 400), (1175, 519)), ((96, 448), (191, 543))]
    print("testMergeBoundingBoxesAcrossFrames-theLabelBoundingBoxArray:",theLabelBoundingBoxArray)
    mergedBoundingBoxes=mergeBoundingBoxesAcrossFrames(labelBoundingBoxCollection, theLabelBoundingBoxArray)
    #print("------------------------------")
    #theLabelBoundingBoxArray= [((192, 400), (335, 543)), ((360, 400), (479, 495)), ((696, 400), (1175, 543))]
    #print("testMergeBoundingBoxesAcrossFrames-theLabelBoundingBoxArray:",theLabelBoundingBoxArray)
    #mergedBoundingBoxes=mergeBoundingBoxesAcrossFrames(labelBoundingBoxCollection, theLabelBoundingBoxArray)

#testProcessImage()
#testMergeBoundingBoxesAcrossFrames()
processProjectVideo()




