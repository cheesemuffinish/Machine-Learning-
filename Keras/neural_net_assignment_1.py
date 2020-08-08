# first neural network with keras make predictions
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# format: [bias, w1, w2]
weights = [-0.2, 0.2, -0.23]


# format: [x1, x2, T/F]
data_set = [[2.78,2.55,0],
        [1.46,2.36,1],
        [3.39,4.40,0],
        [1.38, 1.87,1],
        [3.06,3.03,0],
        [7.62,2.75,1],
        [5.33,2.05,1],
        [6.92,1.7,1],
        [8.67,-0.55,1],
        [7.67,3.51,1]]


def predict(row, weights):
	activation_fx = (weights[1] * row[0]) + (weights[2] * row[1]) + weights[0]
	return 1 if activation_fx >= 0 else 0
	

for row in data_set:
	prediction = predict(row, weights)
	print("expected: %d, predicted: %d" % (row[-1], prediction))