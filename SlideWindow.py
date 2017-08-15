import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

MODELFILENAME='./models/HogClassify.svm'
print('restoring model from: ', MODELFILENAME)
classifier= joblib.load(MODELFILENAME)
classifierPredictions=classifier.predict(X_test)
print('My restored SVM classifier predicts: ', classifierPredictions)
print('For these',numberOfPredictions, 'labels: ', y_test)
print('Test Accuracy of restored classifier = ', round(classifier.score(X_test, y_test), 4))
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', numberOfPredictions,'labels with SVC')
#print("svcPredictions-classifierPredictions:", (svcPredictions-classifierPredictions), ", not all:", not (svcPredictions-classifierPredictions).all())
assert not (svcPredictions-classifierPredictions).all() # subtract equals and assert not all



#image = mpimg.imread('bbox-example-image.jpg')
image = mpimg.imread('test_images/test1.jpg')

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
    print("window_list:", window_list)
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    return window_list

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                       
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
plt.imshow(window_img)
plt.show()
