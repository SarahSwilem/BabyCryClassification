from python_speech_features import mfcc, sigproc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
def gen_features(in_path, out_path, File):
# directory where we your .wav files are
 directoryName = in_path + '/'
 
# directory to put our results in, you can change the name if you like
 resultsDirectory = out_path + '/'

# make a new folder in this directory to save our results in
 if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

# get MFCCs for every .wav file in our specified directory 
 for filename in os.listdir(directoryName):
    if filename.endswith('.wav'): # only get MFCCs from .wavs
        # read in our file
        (sr,sig) = wav.read(directoryName + "/" +filename)
        # get mfcc
        mfcc_feat = mfcc(sig, sr, winlen=0.025, winstep=0.01, numcep=12, nfilt=26, nfft=1024, lowfreq=0, 
                        highfreq=None, preemph=0.97, appendEnergy=True, winfunc=np.hamming)
        mfcc_feat = preprocessing.normalize(mfcc_feat)

        file = open(File, 'w+') # make file/over write existing file
        # create a file to save our results in
        np.savetxt(file, mfcc_feat, delimiter=",") #save MFCCs as .csv
        
        file.close() # close file
        
gen_features('1s_hunger','features',"test.csv") 
gen_features('1s_pain','features',"pain.csv") 
gen_features('1s_normal','features',"normal.csv") 