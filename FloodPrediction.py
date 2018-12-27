import numpy as np
import math, sys, os, copy, random, getopt
import csv

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def DerivertiveSigmoid(x):
    return Sigmoid(x) * (1.0 - Sigmoid(x))

def Tanh(x):
    return np.tanh(x)

def DerivertiveTanh(x):
    return (1.0/np.cosh(x))**2

class NeuralNetwork:
    def __init__(self,LearningRate, MomentumRate, epoch, architect, error,dataFile):
        self.LearningRate = float(LearningRate)
        self.MomentumRate = float(MomentumRate)
        self.epoch = epoch
        self.activation = Tanh
        self.derivertive = DerivertiveTanh
        self.error = float(error)

        #create NeuralNet
        self.initWeight = []
        for i in range(1,len(architect) - 1):
            fanin = architect[i - 1]
            weight = 1/math.sqrt(fanin)
            perceptronNow = architect[i]
            perceptronPrev = architect[i - 1]
            w = np.random.uniform(-1*weight ,weight,[perceptronPrev + 1,perceptronNow + 1])
            self.initWeight.append(w)

        fanin = architect[i - 1]
        weight = 1/math.sqrt(fanin)
        perceptronNowOut = architect[i + 1]
        perceptronPreOut = architect[i] 
        w = np.random.uniform(-1*weight ,weight,[perceptronPreOut + 1,perceptronNowOut])
        self.initWeight.append(w)
    def BackProp(self, train, test):
        b = np.atleast_2d(np.ones(train.shape[0]))
        train = np.concatenate((b.T, train),axis=1)
        self.weightNow = copy.deepcopy(self.initWeight)
        self.weightOld = copy.deepcopy(self.weightNow)
        for k in range(int(self.epoch) + 1):
            for l in range(int(train.shape[0])):
                Alltest = [train[l]]
                Allper = [[]]
                for m in range(len(self.weightNow)):
                    percepLayer = np.dot(Alltest[m], self.weightNow[m])
                    testLayer = self.activation(percepLayer)
                    Allper.append(percepLayer)
                    Alltest.append(testLayer)
                error = test[l] - Alltest[-1]
                grad = [error * self.derivertive(Allper[-1])]

                for n in range(len(Alltest) - 2,0,-1):
                    grad.append(self.derivertive(Alltest[n]) * grad[-1].dot(self.weightNow[n].T)) 
                grad.reverse()

                self.tmpWeightOld = copy.deepcopy(self.weightNow)
                for i in range(len(self.weightNow)):
                    layer = np.atleast_2d(Alltest[i])
                    allGrad = np.atleast_2d(grad[i])
                    deltaW = self.weightNow[i] - self.weightOld[i]


                    self.weightNow[i] += self.MomentumRate * deltaW + self.LearningRate * layer.T.dot(allGrad)

                self.weightOld = copy.deepcopy(self.tmpWeightOld)  

    def Test(self,dataFile,testX,testY,findMin,findMax):
        sse = 0
        correct = 0
        correct10 = 0
        correct01 = 0
        wrong01 = 0
        wrong10 = 0
        index = 0
        arr10 = [1,0]
        arr01 = [0,1]
        for b in testX:
            b = np.concatenate((np.ones(1), np.array(b)))
            for j in range(0, len(self.weightNow)):
                percep = np.dot(b,self.weightNow[j])
                b = self.activation(percep)
            error = testY[index] - b[-1]
            se = 0
            if isinstance(error,np.float64):
                se += error**2
            else: 
                for e in error:
                    se += e**2
            sse += se/2
            classY = testY[index]
            if dataFile == "cross.pat":
                if b[0] > b[1]:
                    out = np.array([1,0])
                else:
                    out = np.array([0,1])
                if (out == testY[index]).all():
                    correct = correct + 1 
                    if(testY[index] == arr10).all():
                        correct01 = correct01 + 1
                    elif(testY[index] == arr01).all():
                        correct01 = correct01 +1 
                elif (out != testY[index]).all():
                    if(testY[index] == arr10).all():
                        wrong10 = wrong10 + 1
                    elif(testY[index] == arr01).all():
                        wrong10 = wrong10 + 1
                out = b 
            else:
                classY = round((classY * (findMax - findMin)) + findMin)
                out = (b * (findMax - findMin)) + findMin 
            if dataFile == 'cross.pat':
                print out, '\t\t',
                print classY
            else:
                print out, '\t\t', 
                print classY
            index = index + 1
        if dataFile == "cross.pat":    
            eev = sse/len(testX)
            # print "\nError = %.8f" % (eev)
            print "Accuracy = %.4f%s" % (correct/(len(testY)*1.0)*100.0,'%')
            print "---------------Confusion Matrix-------------"
            print "----------------Desired Class----------------"
            print "---------[1,0]-------------------[0,1]------"
            print "Pre  [1   {0}                      {1}      ".format(correct01 , wrong10)
            print "Dict  0]                                    "
            print "Class                                       "
            print "     [0   {0}                      {1}      ".format(wrong10 , correct01)
            print "      1]                                    "
            print "--------------------------------------------"
            print "--------------------------------------------"
            print "--------------------------------------------"

        else:
            eev = sse/len(testX)
            eev = (eev * (findMax - findMin)) + findMin
            print "Mean Square Error = %.4f" % (eev)
        return eev

def main(argv):
    # dataset = LoadData('Flood_dataset.csv') #Read Row
    architect = []
    LearningRate = 0.1
    momentumRate = 0.1
    epoch = 100
    error = 0.01
    dataFile = "-"
    try:
        opts, args = getopt.getopt(argv,"chM:L:O:E:F:C")
    except getopt.GetoptError as err:
        print "Wrong Type"
        sys.exit(2)
    print "------------------Start-----------------------"
    for opt,arg in opts:
        if opt in '-M':
            architect = arg.split('-')
            architect = map(int,architect)
            print "Model = {}".format(architect)
        elif opt in '-L':
            LearningRate = arg
            print "LearningRate = " + LearningRate
        elif opt in '-O':
            momentumRate = arg
            print "Momentum rate = {}".format(momentumRate)
        elif opt in '-E':
            epoch = arg
            print  "epoch = {}".format(epoch)
        elif opt in '-F':
            dataFile = arg
            print "File = {}".format(dataFile)
        
    neuralnet = NeuralNetwork(LearningRate,momentumRate,epoch,architect,error,dataFile)

    trainX = []
    trainY = []
    randX = []
    randY = []

    if dataFile == "cross.pat":
        i = 1 
        with open(dataFile,'r') as file:
            for l in file:
                if i % 3 == 0:
                    line = l.split()
                    line = map(int,line)
                    trainY.append(line)
                if i % 3 == 2:
                    line = l.split()
                    line = map(float,line)
                    trainX.append(line)
                i += 1 
        shuffleIndex = random.sample(range(len(trainX)),len(trainY))
        for i in shuffleIndex:
            randX.append(trainX[i])
            randY.append(trainY[i])  
        X = np.array(randX)
        Y = np.array(randY)

    elif dataFile == "Flood-data-set.csv":
        with open(dataFile,'r') as file:
            reader = csv.reader(file,delimiter='\t')
            next(reader)
            next(reader)
            for row in reader:
                line = map(float,row[0:8])
                tline = float(row[8])
                trainX.append(line)
                trainY.append(tline)
        shuffleIndex = random.sample(range(len(trainX)),len(trainY))
        for i in shuffleIndex:
            randX.append(trainX[i])
            randY.append(trainY[i])  
        X = np.array(randX)
        Y = np.array(randY)
        normData = np.column_stack((X,Y))
        findMin = normData.min()
        findMax = normData.max()
        X = (X - findMin) / (findMax - findMin)
        Y = (Y - findMin) / (findMax - findMin)
    crossValid = 1
    if crossValid == 1:
        avgEr = 0.0
        for b in range(0,10):
            print("------------------ Round {} --------------------".format(b))
            print(" Predict\t\t Desired")
            part = int(round(len(trainX)/10.0,0))
            last = (b*part+part) - 1
            if b == 9:
                last = len(trainX) - 1
            tmpX = []
            tmpY = []
            shufX = copy.deepcopy(randX)
            shufY = copy.deepcopy(randY)
            for i in range(last, b*part-1, -1):
                tmpX.append(shufX[i])
                tmpY.append(shufY[i])
                shufX.pop(i)
                shufY.pop(i)
            rTestX = np.array(tmpX)
            rTestY = np.array(tmpY)
            rTrainX = np.array(shufX)
            rTrainY = np.array(shufY)
            if dataFile == "Flood-data-set.csv":
                rTrainX = (rTrainX - findMin) / (findMax - findMin)
                rTrainY = (rTrainY - findMin) / (findMax - findMin)
                rTestX = (rTestX - findMin) / (findMax - findMin)
                rTestY = (rTestY - findMin) / (findMax - findMin)
                neuralnet.BackProp(rTrainX,rTrainY)    
                avgEr = avgEr + neuralnet.Test(dataFile,rTestX,rTestY,findMin,findMax)
            else:
                neuralnet.BackProp(rTrainX,rTrainY)
                avgEr = avgEr + neuralnet.Test(dataFile,rTestX,rTestY,0,0)
            print("\n--------------------- End Round ---------------------\n\n")
        if dataFile == "Flood-data-set.csv":
            print "Average Error = ", avgEr/10
    else:
        if dataFile == "Flood-data-set.csv":
            neuralnet.BackProp(X,Y)
            neuralnet.Test(dataFile,X,Y,findMin,findMax)
        else:
            neuralnet.BackProp(X,Y)
            neuralnet.Test(dataFile,X,Y,0,0)

if __name__ == '__main__':  
    main(sys.argv[1:])