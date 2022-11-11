from random import seed
from random import randrange
from random import random
from random import randint
from csv import reader
from csv import writer
import csv
from types import new_class
import numpy as np
import cv2 
from math import exp, sqrt
import os
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import matplotlib.pyplot as plt 
dataFilename = 'data.csv'
testFilename = 'test.csv'
l_rate = 0.1
n_epoch = 20000
n_hidden = 3
classes = dict()
network = None
lossInfo = [[],[]]
def main():
    mainWindow = Tk() 
    mainWindow.resizable(width=False, height=False) 
    mainWindow.title("Лабораторная работа 4")
    screenWidth = mainWindow.winfo_screenwidth()
    screenHeight = mainWindow.winfo_screenheight()
    x = (screenWidth -400)/ 2
    y = (screenHeight -300)/ 2
    mainWindow.geometry('%dx%d+%d+%d' % (400, 300, x, y))  
    mainWindow.configure(background="#fff")
    #Создание меню
    xCoordinate = 600/2-230/2
    buttonColor="#000"
    buttonWidth="25"
    button1 = Button(text="Выбрать конфиг-файл", command=lambda:process(False),bg=buttonColor,fg="white",width=buttonWidth, height="1",font=("MS Sans Serif", 14),relief="groove",activebackground="#dc3f3f")
    button1.place(x=xCoordinate / 2 - 30, y=90)
    button2 = Button(text="Предсказать по изображению", command=lambda:predictByImage(),bg=buttonColor,fg="white",width=buttonWidth, height="1",font=("MS Sans Serif", 14),relief="groove",activebackground="#dc3f3f")
    button2.place(x=xCoordinate / 2 - 30, y=140)
    # process(True)
    mainWindow.mainloop() 
    # while (True):
    #     print("1. Predict\n2. Train")
    #     inp = input()
    #     if inp == 1:
    #         predictByImage()
    #     if inp == 2:
    #         process(False)

#Open file
def openFile(isDefault):
    global isNew
    global trainData
    global train_y_trues
    if isDefault==True:
        path=os.path.dirname(os.path.abspath(__file__))+'\\config.txt'
    else:
        path = filedialog.askopenfilename(filetypes=(("Текстовый файл", "*.txt"),("Текстовый файл", "*.txt")))
    if path!='':
        images = []
        lines = open(path).readlines()
        files = []
        j = 0
        for i in range(3):
            files.append([])
            while  j < len(lines) and lines[j].__contains__("-----") == False:
                files[i].append(lines[j].replace("\n",""))
                j += 1
            j += 1
        train_y_trues = []
        for i in range(len(files[0])):
            if files[0][i].__contains__(".bmp"):
                images.append(binarize(cv2.imread(files[0][i], cv2.IMREAD_COLOR)))
                train_y_trues.append([])
                if files[0][i].__contains__("flesh"):
                    images[-1].append("flesh")
                if files[0][i].__contains__("marker"):
                    images[-1].append("marker")
                if files[0][i].__contains__("pencil"):
                    images[-1].append("pencil")
        trainData = np.array(images)
        with open("data.csv", 'w', newline='') as myfile:
            wr = writer(myfile, quoting=csv.QUOTE_ALL)
            for row in trainData:
                wr.writerow(row)
        images = []
        for i in range(len(files[1])):
            if files[1][i].__contains__(".bmp"):
                images.append(binarize(cv2.imread(files[1][i],cv2.IMREAD_COLOR)))
                if files[1][i].__contains__("flesh"):
                    images[-1].append("flesh")
                if files[1][i].__contains__("marker"):
                    images[-1].append("marker")
                if files[1][i].__contains__("pencil"):
                    images[-1].append("pencil")
        testData = np.array(images)
        with open("test.csv", 'w', newline='') as myfile:
            wr = writer(myfile, quoting=csv.QUOTE_ALL)
            for row in testData:
                wr.writerow(row)
        isNew = True
        return files[2][0]

# Binarize image
def binarize(img):
    vectorImg = []
    img = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(3):
                if img[i][j][k] > 245:
                    vectorImg.append(0)
                else:
                    vectorImg.append(1)
    dataRow = []
    rowSum = 0
    for i in range(len(vectorImg)):
        rowSum += vectorImg[i]
        if i != 0 and i % 23 == 0:
            dataRow.append(rowSum/24)
            rowSum = 0
    dataRow.append(sum(vectorImg)/len(vectorImg))
    return dataRow

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column, lookup = None):
    if lookup == None:
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate, beta = 0.9):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
            if i != 1:
                oldInputs = inputs
                inputs = []
                for j in range(len(oldInputs)):
                    current = oldInputs[j]
                    previousInputs = [neuron['output'] for neuron in network[i - 2]]
                    sumIn = 0
                    for item in previousInputs:
                        sumIn += item * current
                    inputs.append(sumIn)
                if i != 2:
                    oldInputs = inputs
                    inputs = []
                    for j in range(len(oldInputs)):
                        current = oldInputs[j]
                        previousInputs = [neuron['output'] for neuron in network[i - 3]]
                        sumIn = 0
                        for item in previousInputs:
                            sumIn += item * current
                        inputs.append(sumIn)
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['squares'][j] = beta * neuron['squares'][j] + (1 - beta) * neuron['delta'] ** 2
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j] / (sqrt(neuron['squares'][j]) + pow(1,-10))
            neuron['squares'][-1] = beta * neuron['squares'][-1] + (1 - beta) * neuron['delta'] ** 2
            neuron['weights'][-1] -= l_rate * neuron['delta'] / (sqrt(neuron['squares'][-1]) + pow(1,-10))
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    global lossInfo
    lossInfo = [[],[]]
    for epoch in range(n_epoch):
        miniBatch = getMiniBatchItems(len(train),int(0.2*len(train)))
        for rowIndex in miniBatch:
            row = train[rowIndex]
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        if epoch % 10 == True:
            pred = []
            true = []
            for item in train:
                pred.append(predict(network, item))
                true.append(item[-1])
            lossInfo[0].append(epoch)
            lossInfo[1].append(mseLoss(true, pred))
    fig, ax = plt.subplots()
    ax.plot(lossInfo[0],lossInfo[1],'r')
    plt.title('Функция потерь')
    plt.show()
def mseLoss(y_true, y_pred):
    sum = 0.0
    for i in range(len(y_true)):
        sum += (y_true[i] - y_pred[i]) ** 2
    return sum/len(y_true) 

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    global network
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)], 'squares':[0 for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    second_hidden_layer = [{'weights':[random() for i in range(len(hidden_layer) + 1)], 'squares': [0 for i in range(len(hidden_layer) + 1)]} for i in range(n_hidden)]
    network.append(second_hidden_layer)
    third_hidden_layer = [{'weights':[random() for i in range(len(second_hidden_layer) + 1)], 'squares': [0 for i in range(len(second_hidden_layer) + 1)]} for i in range(n_hidden)]
    network.append(third_hidden_layer)
    output_layer = [{'weights':[random() for i in range(len(third_hidden_layer) + 1)], 'squares': [0 for i in range(len(third_hidden_layer) + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
 
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
 
# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train,l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	return network

#get indexes for mini batch
def getMiniBatchItems(all, count):
    items = []
    while len(items) < count:
        item = randint(0,all-1)
        if items.__contains__(item) == False:
            items.append(item)
    return items 

# Predict by selected image
def predictByImage():
    path = filedialog.askopenfilename(filetypes=(("Изображение", "*.bmp"),("Изображение", "*.bmp")))
    if path != '':
        image = binarize(cv2.imread(path, cv2.IMREAD_COLOR))
        image.append(None)
        predicted = predict(network, image)
        messagebox.showinfo(title="результат", message=list(classes.keys())[list(classes.values()).index(predicted)])

# load and prepare data
def process(isDefault):
    global classes
    resultsPath = openFile(isDefault)
    dataset = load_csv(dataFilename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    classes = str_column_to_int(dataset, len(dataset[0])-1)
    testset = load_csv(testFilename)
    for i in range(len(testset[0])-1):
        str_column_to_float(testset, i)
    test_trues = []
    for testRow in testset:
        test_trues.append(testRow[-1])
    str_column_to_int(testset, len(testset[0])-1, classes)
    network = back_propagation(dataset, l_rate, n_epoch, n_hidden)
    predictions = list()
    for row in testset:
        prediction = predict(network, row)
        predictions.append(prediction)
    if resultsPath == None:
        resultsPath = "results.txt"
    f = open(resultsPath , 'w')
    resultStr = "true    predicted\n"
    for i in range(len(test_trues)):
        resultStr += '%-8s' % test_trues[i]
        resultStr += '%-8s' % list(classes.keys())[list(classes.values()).index(predictions[i])]
        resultStr += "\n"
    f.write(resultStr)
    f.close()
main()