from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy
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
        (SamplingRate,sig) = wav.read(directoryName + "/" +filename)
        

        # get mfcc
        mfcc_feat = mfcc(sig,SamplingRate)

        # get filterbank energies
        fbank_feat = logfbank(sig,SamplingRate)
        file = open(File, 'w+') # make file/over write existing file
        # create a file to save our results in
        numpy.savetxt(file, fbank_feat, delimiter=",") #save MFCCs as .csv
        
        file.close() # close file
        
gen_features('test','features',"test.csv") 
gen_features('1s_pain','features',"pain.csv") 
gen_features('1s_normal','features',"normal.csv") 