import numpy as np
import scipy.io.wavfile as wav
import pickle
from sklearn.neural_network import MLPClassifier

def normalizeAudio(audio):
	return audio.astype(float) / audio.max()

def chunkAudio(audio, chunk_size, offset):
	results = []
	startingIndex = 0
	while startingIndex + chunk_size < len(audio):
		results.append(audio[startingIndex : startingIndex + chunk_size])
		startingIndex += offset
	return results

#Read in wav data
#audioFilesData is a list of tuples containing the audio file data and the corresponding classification
audioFilesData = []
audioFilesData.append((wav.read("EthanTest.wav"), [1,0]))
audioFilesData.append((wav.read("EthanTest2.wav"), [1,0]))
audioFilesData.append((wav.read("KatieTest.wav"), [0,1]))
audioFilesData.append((wav.read("KatieTest2.wav"), [0,1]))

audioRate = audioFilesData[0][0][0]

#Extract the voice data and normalize the voice
print("Extracting and Normalizing Voice Data")

#iterate through each "row" in audioFilesData and normalize the audio data on scale from -1 to 1 and keep the classification
audioFilesData = [(normalizeAudio(audioFilesData[i][0][1]), audioFilesData[i][1]) for i in range(len(audioFilesData))]


ethanSampleVoice = wav.read("EthanSample.wav")[1]
katieSampleVoice = wav.read("KatieSample.wav")[1]

ethanSampleVoice = normalizeAudio(ethanSampleVoice)
katieSampleVoice = normalizeAudio(katieSampleVoice)

#size of each audio frame to analyze
#audioRate is number of samples/sec ... each frame should be .01 of a second
frameLength = int(0.1 * audioRate)

#Create the training set for the computer
print("Creating the Training Set")
trainingSet = []
targetSet = []

#Get the audio data in audioFilesData and split the data into frames of `frameLength` size
for data in audioFilesData:
	voice = data[0]
	classification = data[1]
	for chunk in chunkAudio(voice, frameLength, frameLength/3):
		trainingSet.append(chunk)
		targetSet.append(classification)

print("Training MLP")
mlp = MLPClassifier(hidden_layer_sizes=(frameLength/2), max_iter=100, verbose=True)
mlp.fit(trainingSet, targetSet)

print("Checking Results")

print("Ethan Results:")

ethanResults = []
katieResults = []

for i in range(ethanSampleVoice.size / frameLength):
	startingIndex = frameLength * i
	ethanResults.append(mlp.predict(np.array(ethanSampleVoice[startingIndex : startingIndex + frameLength]).reshape(1, -1)))

print(np.sum(ethanResults, 0))

print("Katie Results:")

for i in range(katieSampleVoice.size / frameLength):
	startingIndex = frameLength * i
	katieResults.append(mlp.predict(np.array(katieSampleVoice[startingIndex : startingIndex + frameLength]).reshape(1, -1)))

print(np.sum(katieResults, 0))

#Save Model
pickle.dump(mlp, open("voice_model.sav", "wb"))
