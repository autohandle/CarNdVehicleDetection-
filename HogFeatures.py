from skimage.feature import hog

import numpy as np

# https://github.com/scikit-image/scikit-image/issues/1889
#In the meantime, as a workaround you could simply make your images exclusively positive
#with an offset before calling this. As it's based on gradients you won't get into trouble;
#just subtract the minimum value of the input array from the array in the call.

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    #print("get_hog_features-max(img):",np.max(img),", min(img):", np.min(img))
    imageMinimum=np.min(img)
    if imageMinimum<0 :
      imageCopy=img.copy()-np.min(img)
    else:
      imageCopy=img.copy()
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(imageCopy, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(imageCopy, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        #print("get_hog_features-max(features):",np.max(features),", min(features):", np.min(features))
        return features
