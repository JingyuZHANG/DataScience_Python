import matplotlib.pyplot as plt
import math
import numpy
import pandas
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import initializations

# LSTM for a regression framing
# convert an array of values into a dataset matrix

# fix random seed for reproducibility
numpy.random.seed(10)
initValue1 = 0
initValue2 = 0
#print(1000 * numpy.random.random_sample())
checker1 = False
checker2 = False

bestScore = 10000000
bestInitValue1 = -1
bestInitValue2 = -1

def get_scale():
    return random.uniform(0, 1000)


def my_init1(shape, name="Random1"):

    global checker1
    if checker1 is False:
        checker1 = True
        global initValue1
        initValue1 = get_scale()
        print("initValue1:" + str(initValue1))
    return initializations.normal(shape, scale=initValue1, name=name)


def my_init2(shape, name="Random2"):
    global initValue2
    global checker2
    if checker2 is False:
        checker2 = True
        initValue2 = get_scale()
        print("initValue2:" + str(initValue2))
    return initializations.normal(shape, scale=initValue2, name=name)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def runRnnlstm(trainX, testX,trainY,testY):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(2, init=my_init1, input_dim=1, return_sequences=True))
    model.add(LSTM(2, init=my_init2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=10, batch_size=1, verbose=False)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    return trainScore

# load the dataset
dataframe = pandas.read_csv('cafe.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

for index in range(0, 10):
    print("==============================")
# training the model
    trainScore = runRnnlstm(trainX, testX,trainY,testY)
    checker1 = False
    checker2 = False
    print("taScore," + str(trainScore))
    if bestScore > trainScore:
        print("improve")
        bestScore = trainScore
        bestInitValue1 = initValue1
        bestInitValue2 = initValue2
    print("This is the best trainScore in, index," + str(index)  + "," + str(bestScore) + "," + str(trainScore) + ","
          + str(bestInitValue1) + "," + str(bestInitValue2))
# shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
    #plt.show()