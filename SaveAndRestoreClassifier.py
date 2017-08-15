from sklearn.externals import joblib
import time
from sklearn.preprocessing import StandardScaler


timeString=time.strftime("%Y%m%d-%H%M%S")
MODELFILENAME='./models/Classifier-'+timeString+".svm"
print("SaveAndRestoreClassifier-MODELFILENAME:", MODELFILENAME)
SCALERFILENAME='./models/Scaler-'+timeString+".svm"
print("SaveAndRestoreClassifier-SCALERFILENAME:", SCALERFILENAME)

# save the model
def saveClassifier(classifier, file=MODELFILENAME):
    print('saveClassifier-saving model to: <'+file+">")
    joblib.dump(classifier, file)

def restoreClassifier(file=MODELFILENAME):
    print('restoreClassifier-restoring model from: <'+file+">")
    classifier= joblib.load(file)
    return classifier

# save the scaler
import pickle

def XsaveScaler(scaler, file=SCALERFILENAME):
    print('saveScaler-saving scaler to: <'+file+">")
    #joblib.dump(scaler, file) 
    f = open(file, 'wb')
    pickle.dump(scaler, f)
    f.close()

def XrestoreScaler(file=SCALERFILENAME):
    print('restoreScaler-restoring scaler from: <'+file+">")
    scaler = joblib.load(file)
    f = open(file, 'rb')
    scaler = pickle.load(f)
    f.close()
    return scaler

def saveScaler(scaler, file=SCALERFILENAME):
    print('saveScaler-scaler: ', scaler, ", get_params:", scaler.get_params(deep=True), ", mean:", scaler.mean_, ", std:", scaler.std_)
    print('saveScaler-saving scaler to: <'+file+">")
    joblib.dump(scaler, file) 

def saveScalerFitX(X, file=SCALERFILENAME):
    print('saveScalerFitX-saving X to: <'+file+">")
    joblib.dump(X, file) 

def restoreScaler(file=SCALERFILENAME):
    print('restoreScaler-restoring scaler from: <'+file+">")
    scaler = joblib.load(file)
    print('restoreScaler-scaler: ', scaler, ", get_params:", scaler.get_params(deep=True), ", mean:", scaler.mean_, ", std:", scaler.std_)
    return scaler

def restoreScalerFromX(file=SCALERFILENAME):
    print('restoreScaler-restoring scaler from: <'+file+">")
    X = joblib.load(file)
    X_scaler=StandardScaler().fit(X)
    return X_scaler