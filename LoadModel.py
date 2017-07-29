import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
import scipy.io.wavfile as wav
import sys

def normalize(data):
	return data.astype(float) / data.max()


print("Reading in Audio File...")
audio = wav.read(sys.argv[2])
rate = audio[0]
data = normalize(audio[1])

frameSize = int(0.1 * rate)

print("Loading MLP Model...")
model = pickle.load(open(sys.argv[1], "rb"))
print("Running Prediction...")
results = []
for i in range(data.size / frameSize):
	startingIndex = i * frameSize
	results.append(model.predict(np.array(data[startingIndex : startingIndex + frameSize]).reshape(1, -1)))

print(results)
print(np.sum(results, 0))
