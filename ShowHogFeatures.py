import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import HogFeatures
import FeatureVectorConfig

# features, hog_image=def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

imageNames=["./training/smallset/vehicles_smallset/cars1/17.jpeg", "./training/smallset/non-vehicles_smallset/notcars1/extra08.jpeg"]

figureColumnCount=2
figureRowCount=len(imageNames)
showImages = plt.figure(figsize = (figureColumnCount*5,figureRowCount*5))

for imageName, imageIndex in zip(imageNames, range(0,len(imageNames))):

    print("imageName:", imageName, ", imageIndex:", imageIndex)
    img = mpimg.imread(imageName)

    features, hogImage=HogFeatures.get_hog_features(img[:,:,0], # red
        FeatureVectorConfig.ORIENTATIONBINS,
        FeatureVectorConfig.PIXELSPERCELL,
        FeatureVectorConfig.CELLSPERBLOCK,
        vis=True, feature_vec=True)


    p=showImages.add_subplot(figureRowCount,figureColumnCount,imageIndex*figureColumnCount+1)
    p.set_title(imageName+"\nimage", fontsize=10)
    p.imshow(img)

    p=showImages.add_subplot(figureRowCount,figureColumnCount,imageIndex*figureColumnCount+2)
    p.set_title(imageName+"\nHOG image", fontsize=10)
    p.imshow(hogImage, cmap='gray')

    plt.tight_layout()

plt.show()
showImages.savefig('./output_images/hogFeatures/HogFeatures.png')
plt.close()
