from numpy import array
from random import random
from math import sin, sqrt
import math
import numpy
import pandas


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import initializations

numpy.random.seed(10)

iter_max = 20
pop_size = 2
dimensions = 2
c1 = 3
c2 = 3
err_crit = 0.000000001

class Particle:
    pass

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
     
def normVale(scal):
    if scal[0] <= 0:
        scal[0] = random()
    if scal[1] <= 0:
        scal[1] = random()   
    return [scal[0], scal[1]]
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def runRnnlstm(trainX, testX,trainY,testY,scal):
    # create and fit the LSTM network
    model = Sequential()
    scale = normVale(scal)
    model.add(LSTM(2, init=lambda shape, name: initializations.normal(shape, scale=scale[0] , name="Layer1"), input_dim=1, return_sequences=True))
    model.add(LSTM(2, init=lambda shape, name: initializations.normal(shape, scale=scale[1] , name="Layer2")))
    print("==============================")
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=False)

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
    return (1000-trainScore)


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
 
#initialize the particles
particles = []
for i in range(pop_size):
    p = Particle()
    p.params = array([random()*1000 for i in range(dimensions)])
    p.fitness = 0.0
    p.v = 0.0
    particles.append(p)

# let the first particle be the global best
gbest = particles[0]
err = 999999999
while i < iter_max :
#    for p in particles (Crossover with other particles):
    if (i%5==0 and i!=0):
            #p1   
        fitness = runRnnlstm(trainX, testX,trainY,testY,[particles[0].params[0],particles[1].best[1]])
        if fitness > particles[0].fitness:
            particles[0].fitness = fitness
            particles[0].best = [particles[0].params[0],particles[1].best[1]]
    
        if fitness > gbest.fitness:
            gbest = particles[0]
            gbest.params = particles[0].best
        v1 = particles[0].v + c1 * random() * (particles[0].best - particles[0].params) \
                + c2 * random() * (gbest.params - particles[0].params)
        particles[0].params = particles[0].params + v1
    
        #p2
        fitness = runRnnlstm(trainX, testX,trainY,testY,[particles[0].best[0], particles[1].params[1]]) #particles[1].params)
        if fitness > particles[1].fitness:
            particles[1].fitness = fitness
            particles[1].best = [particles[0].best[0], particles[1].params[1]]
        if fitness > gbest.fitness:
            gbest = particles[1]
        v2 = particles[1].v + c1 * random() * (particles[1].best - particles[1].params) \
                + c2 * random() * (gbest.params - particles[1].params)
        particles[1].params = particles[1].params + v2  
    else: 
        #p1   
        fitness = runRnnlstm(trainX, testX,trainY,testY,particles[0].params)
        if fitness > particles[0].fitness:
            particles[0].fitness = fitness
            particles[0].best = particles[0].params
    
        if fitness > gbest.fitness:
            gbest = particles[0]
        v1 = particles[0].v + c1 * random() * (particles[0].best - particles[0].params) \
                + c2 * random() * (gbest.params - particles[0].params)
        particles[0].params = particles[0].params + v1
    
        #p2
        fitness = runRnnlstm(trainX, testX,trainY,testY,particles[1].params)
        if fitness > particles[1].fitness:
            particles[1].fitness = fitness
            particles[1].best = particles[1].params
        if fitness > gbest.fitness:
            gbest = particles[1]
        v2 = particles[1].v + c1 * random() * (particles[1].best - particles[1].params) \
                + c2 * random() * (gbest.params - particles[1].params)
        particles[1].params = particles[1].params + v2   
    print("gbest" + gbest.fitness)
    i  += 1

## Uncomment to print particles
#for p in particles:
#    print 'params: %s, fitness: %s, best: %s' % (p.params, p.fitness, p.best)