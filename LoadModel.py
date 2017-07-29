import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
import scipy.io.wavfile as wav
import sys

def normalize(data):
	return data.astype(float) / data.max()

def chunkAudio(audio, chunk_size, offset):
	results = []
	startingIndex = 0
	while startingIndex + chunk_size < len(audio):
		results.append(audio[startingIndex : startingIndex + chunk_size])
		startingIndex += offset
	return results


print("Reading in Audio File...")
audio = wav.read(sys.argv[2])
rate = audio[0]
data = normalize(audio[1])

frameSize = int(0.1 * rate)

print("Loading MLP Model...")
model = pickle.load(open(sys.argv[1], "rb"))
print("Running Prediction...")
results = []
for chunk in chunkAudio(data, frameSize, frameSize / 3):
	results.append(model.predict(np.array(chunk).reshape(1, -1)))

print(results)
print(np.sum(results, 0))
