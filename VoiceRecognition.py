import numpy as np
import scipy.io.wavfile as wav
import pickle
from sklearn.neural_network import MLPClassifier

def normalizeAudio(audio):
	return audio.astype(float) / audio.max()

#Read in wav data
ethanTestData = wav.read("EthanTest2.wav")
katieTestData = wav.read("KatieTest2.wav")

#Extract the voice data
print("Getting Voice Data")
ethanTestVoice = ethanTestData[1]
katieTestVoice = katieTestData[1]

ethanSampleVoice = wav.read("EthanSample.wav")[1]
katieSampleVoice = wav.read("KatieSample.wav")[1]

audioRate = ethanTestData[0]

#Normalize audio on scale from -1 to 1
print("Normalizing Data")
ethanTestVoice = normalizeAudio(ethanTestVoice)
katieTestVoice = normalizeAudio(katieTestVoice)
ethanSampleVoice = normalizeAudio(ethanSampleVoice)
katieSampleVoice = normalizeAudio(katieSampleVoice)

#size of each audio frame to analyze
#audioRate is number of samples/sec ... each frame should be .01 of a second
frameLength = int(0.1 * audioRate)

#Create the training set for the computer
print("Creating the Training Set")
trainingSet = []
targetSet = []
for i in range(ethanTestVoice.size / frameLength):
	startingIndex = frameLength * i
	trainingSet.append(ethanTestVoice[startingIndex : startingIndex + frameLength])
	targetSet.append([1, 0])

for i in range(katieTestVoice.size / frameLength):
	startingIndex = frameLength * i
	trainingSet.append(katieTestVoice[startingIndex : startingIndex + frameLength])
	targetSet.append([0, 1])

print("Training MLP")
mlp = MLPClassifier(hidden_layer_sizes=(frameLength/2), max_iter=100, verbose=True)
mlp.fit(trainingSet, targetSet)

print("Checking Results")

print("Ethan Results:")

ethanResults = []
katieResults = []

for i in range(ethanSampleVoice.size / frameLength):
	startingIndex = frameLength * i
	ethanResults.append(mlp.predict(ethanSampleVoice[startingIndex : startingIndex + frameLength]))

print(np.sum(ethanResults, 0))

print("Katie Results:")

for i in range(katieSampleVoice.size / frameLength):
	startingIndex = frameLength * i
	katieResults.append(mlp.predict(katieSampleVoice[startingIndex : startingIndex + frameLength]))

print(np.sum(katieResults, 0))

#Save Model
pickle.dump(mlp, open("voice_model.sav", "wb"))
