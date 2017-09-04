COLORSPACE = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENTATIONBINS = 9
PIXELSPERCELL = 8
CELLSPERBLOCK = 2
HOGCHANNEL = 'ALL' # Can be 0, 1, 2, or "ALL"


# from WindowSearch
### TODO: Tweak these parameters and see how the results change.
#color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#orient = 9  # HOG orientations
#pix_per_cell = 8 # HOG pixels per cell
#cell_per_block = 2 # HOG cells per block
#hog_channel = 0 # Can be 0, 1, 2, or "ALL"

SPATIALSIZE = (16, 16) # Spatial binning dimensions
HISTOGRAMBINS = 16    # Number of histogram bins
SPATIALFEATURES = True # Spatial features on or off
HISTOGRAMFEATURES = True # Histogram features on or off
HOGFEATURES = True # HOG features on or off
