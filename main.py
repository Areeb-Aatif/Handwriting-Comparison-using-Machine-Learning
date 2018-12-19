from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook
from keras.utils import np_utils
import matplotlib.pyplot as plt

RawData = []
RawTarget = []
TrainingData = []
TrainingTarget = []
ValidationData = []
ValidationTarget = []
TestingData = []
TestingTarget = []

# Reads data from the concatenation and subtraction dataset.
def GenerateRawData(filePath):    
    dataMatrix = [] 
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            dataRow = []
            for column in row[2:-1]:
                dataRow.append(int(column))
            dataMatrix.append(dataRow)
    print("Raw Data Generated.")
    return dataMatrix

# Reads dataset from csv and retrieve target values for the input.
def GenerateTargetVector(filePath):
    targetVector = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:  
            targetVector.append(int(row[-1]))
    print("Raw Target Generated.")
    return targetVector

# Dividing our input data matrix to training data.
def GenerateTrainingDataMatrix(RawData, TrainingPercent):
    T_len = int(math.ceil(len(RawData)*0.01*TrainingPercent))
    TrainingData = RawData[:T_len]
    print(str(TrainingPercent) + "% Training Data Generated.")
    return TrainingData

# Dividing our target vector to training target.
def GenerateTrainingTargetVector(RawTarget, TrainingPercent):
    TrainingLen = int(math.ceil(len(RawTarget)*(TrainingPercent*0.01)))
    TrainingTarget = RawTarget[:TrainingLen]
    print(str(TrainingPercent) + "% Training Target Generated.")
    return TrainingTarget

# Dividing our input data matrix to validation data.
def GenerateValidationDataMatrix(RawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(RawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    ValidationData = RawData[TrainingCount:V_End]
    print (str(ValPercent) + "% Validation Data Generated.")  
    return ValidationData

# Dividing our target vector to validation target.
def GenerateValidationTargetVector(RawTarget, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(RawTarget)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    ValidationTarget = RawTarget[TrainingCount:V_End]
    print (str(ValPercent) + "% Validation Target Generated.")
    return ValidationTarget

# Dividing our input data matrix to testing data.
def GenerateTestingDataMatrix(RawData, TestPercent, Count): 
    testSize = int(math.ceil(len(RawData)*TestPercent*0.01))
    T_End = Count + testSize
    TestingData = RawData[Count:T_End]
    print (str(TestPercent) + "% Testing Data Generated.")  
    return TestingData

# Dividing our target vector to testing target.
def GenerateTestingTargetVector(RawTarget, TestPercent, Count): 
    testSize = int(math.ceil(len(RawTarget)*TestPercent*0.01))
    T_End = Count + testSize
    TestingTarget = RawTarget[Count:T_End]
    print (str(TestPercent) + "% Testing Target Generated.")
    return TestingTarget

# This function computes BigSigma matrix of dimensions same as the number of features.
# It computes variance for each of the features and stores it in BigSigma matrix.
def GenerateBigSigma(TrainingData, MuMatrix):
    
    BigSigma = np.zeros((len(TrainingData[0]), len(TrainingData[0])))
    TrainingDataT = np.transpose(TrainingData)

    VarVector = []
    for i in range(0, len(TrainingDataT)):
        Vector = []
        for j in range(0, len(TrainingData)):
            Vector.append(TrainingDataT[i][j])
        VarVector.append(np.var(Vector))

    # Storing variance at only the diagonal terms. Constraining Sigma to be a diagonal matrix.
    for j in range(len(TrainingDataT)):
        BigSigma[j][j] = VarVector[j]

    BigSigma = np.dot(200, BigSigma)

    return BigSigma

# Function to find the PHI matrix.
def GetPhiMatrix(Data, MuMatrix, BigSigma):

    BigSigmaInv = np.linalg.pinv(BigSigma)
    Phi = np.zeros((len(Data), len(MuMatrix)))

    for i in range(0, len(MuMatrix)):
        for j in range(0, len(Data)):
            Phi[j][i] = GetRadialBasis(Data[j], MuMatrix[i], BigSigmaInv)

    return Phi

# This function multiplies the scalar value obtained from GetScalar function with -0.5
# and then takes the exponential value to generate PHI for a particular input index.
def GetRadialBasis(DataRow, MuRow, BigSigmaInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow, MuRow, BigSigmaInv))
    return phi_x

# This function converts input vector x to a specific scalar value.
def GetScalar(DataRow, MuRow, BigSigInv):  
    R = np.subtract(DataRow, MuRow)
    T = np.dot(BigSigInv, np.transpose(R))  
    L = np.dot(R, T)
    return L

# This function calculates weight vector w using the formula for closed form solution 
# with least squared regularization.
def GetWeightsClosedForm(PHI, T, Lambda = 0.01):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    
    return W

# This function outputs Y for a particular value of x,w (Linear Regression). 
def GetValTest(Data_PHI, W):
    Y = np.dot(W, np.transpose(Data_PHI))
    return Y

# This function calculates Erms for a particular set of data. 
def GetErms(Data_TEST_OUT, DataAct):

    sum = 0.0
    accuracy = 0.0
    counter = 0

    for i in range (0,len(Data_TEST_OUT)):
        sum = sum + np.power((DataAct[i] - Data_TEST_OUT[i]),2)
        if(int(np.around(Data_TEST_OUT[i], 0)) == DataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(Data_TEST_OUT)))

    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(Data_TEST_OUT))))

# Stochastic Gradient Descent optimization for linear regression.
def SGD_LinearRegression(W, TrainingPhi, ValidationPhi, TestingPhi, TrainingTarget, ValidationTarget, TestingTarget, Lamda, LearningRate):

    W_Now = np.dot(220, W)

    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = [] 

    for i in range(0, 500):

        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TrainingPhi[i])),TrainingPhi[i])
        La_Delta_E_W  = np.dot(Lamda, W_Now)
        Delta_E       = np.add(Delta_E_D, La_Delta_E_W)

        Delta_W       = -np.dot(LearningRate, Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next

        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TrainingPhi, W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT, TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(ValidationPhi, W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT, ValidationTarget)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TestingPhi, W_T_Next) 
        Erms_Test     = GetErms(TEST_OUT, TestingTarget)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))

    print ('\nUBITname      = *****')
    print ('Person Number = *****')
    print ('---------------------------------------------------------')
    print ('--------------Linear Regression------------------------')
    print ('---------------------------------------------------------')
    print ("Number of Clusters = 10 \nLambda  = " + str(Lamda) +"\nlearning Rate = "+ str(LearningRate))
    print ("E_rms Training   = " + str(min(L_Erms_TR)))
    print ("E_rms Validation = " + str(min(L_Erms_Val)))
    print ("E_rms Testing    = " + str(min(L_Erms_Test)))

# Calculating the sigmoid value for logistic regression.
def GetSigmoid(Y):

    return (1 / (1 + np.exp(-Y)))

# Calculating the accuracy of the logistic regression model by comparing the predicted values
# to the target values.
def LossFunction(Sigmoid, Target):

    Observations = len(Target)

    Cost1 = -np.dot(np.transpose(Target), np.log(Sigmoid))

    Cost2 = -np.dot(np.transpose(np.subtract(1, Target)), np.log(np.subtract(1, Sigmoid)))

    Cost = (Cost1 + Cost2) / Observations

    return Cost

# We classify our sigmoid function into two class. For a value > 0.5 we set its target to 1 
# and when our value is < 0.5 we set its target to 0.
def Classify(Predicted):
    l = []
    for c in Predicted:
        if(c >= 0.5):
            l.append(1)
        else:
            l.append(0) 
    return l

# Finding the accuracy for the predicted values using sigmoid function.
def Accuracy(Predicted, Actual):
    diff = np.array(Predicted) - np.array(Actual)
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

# Stochastic Gradient Descent solution for the logistic regression
def SGD_LogisticRegression(W, TrainingPhi, ValidationPhi, TestingPhi, TrainingTarget, ValidationTarget, TestingTarget, Lamda, LearningRate):

    W_Now = np.dot(10, W)

    ValCost   = []
    TRCost    = []
    TestCost  = []
    TRAccuracy = []
    ValAccuracy = []
    TestAccuracy = []

    for i in range(0, 50):

        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TrainingPhi[i])),TrainingPhi[i])
        La_Delta_E_W  = np.dot(Lamda, W_Now)
        Delta_E       = np.add(Delta_E_D, La_Delta_E_W)

        Delta_W       = -np.dot(LearningRate, Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next

        #-----------------TrainingData Accuracy---------------------#
        TR_OUT = GetValTest(TrainingPhi, W_T_Next) 
        Sigmoid_Tr = GetSigmoid(TR_OUT)
        Tr_Cost = LossFunction(Sigmoid_Tr, TrainingTarget)
        TRCost.append(float(Tr_Cost))
        Predicted = Classify(Sigmoid_Tr)
        Acc = Accuracy(Predicted, TrainingTarget)
        TRAccuracy.append(Acc)
        print(Tr_Cost)
        print(Acc)
        
        #-----------------ValidationData Accuracy---------------------#
        VAL_OUT = GetValTest(ValidationPhi, W_T_Next) 
        Sigmoid_Val = GetSigmoid(VAL_OUT)
        Val_Cost = LossFunction(Sigmoid_Val, ValidationTarget)
        ValCost.append(float(Val_Cost))
        Predicted = Classify(Sigmoid_Val)
        Acc = Accuracy(Predicted, ValidationTarget)
        ValAccuracy.append(Acc)
        
        #-----------------TestingData Accuracy------------------------#
        TEST_OUT = GetValTest(TestingPhi, W_T_Next) 
        Sigmoid_Test = GetSigmoid(TEST_OUT)
        Test_Cost = LossFunction(Sigmoid_Test, TestingTarget)
        TestCost.append(float(Test_Cost))
        Predicted = Classify(Sigmoid_Test)
        Acc = Accuracy(Predicted, TestingTarget)
        TestAccuracy.append(Acc)

    print ('\nUBITname      = *****')
    print ('Person Number = *****')
    print ('---------------------------------------------------------')
    print ('---------------Logistic Regression-----------------------')
    print ('---------------------------------------------------------')
    print ("Number of Clusters = 10 \nLambda  = "+ str(Lamda) + "\nLearning Rate = " + str(LearningRate))
    print ("Training Cost       = " + str(min(TRCost)))
    print ("Validation Cost     = " + str(min(ValCost)))
    print ("Testing Cost        = " + str(min(TestCost)))
    print ("Training Accuracy   = " + str(max(TRAccuracy)*100))
    print ("Validation Accuracy = " + str(max(ValAccuracy)*100))
    print ("Testing Accuracy    = " + str(max(TestAccuracy)*100))

# Main function to read the input data matrix from the csv file and divide it into training, 
# validation and testing dataset.
def ReadData(filename):

    global ReadData, RawTarget, TrainingData, TrainingTarget, ValidationData
    global ValidationTarget, TestingData, TestingTarget
    
    # head = filename.replace('.csv', '')
    # head = head.split('/')[1]

    RawData = GenerateRawData(filename)
    RawTarget = GenerateTargetVector(filename)

    TrainingData = GenerateTrainingDataMatrix(RawData, 80)
    TrainingTarget = GenerateTrainingTargetVector(RawTarget, 80)

    ValidationData = GenerateValidationDataMatrix(RawData, 10, len(TrainingData))
    ValidationTarget = GenerateValidationTargetVector(RawTarget, 10, len(TrainingTarget))

    TestingData = GenerateTestingDataMatrix(RawData, 10, len(TrainingData) + len(ValidationData))
    TestingTarget = GenerateTestingTargetVector(RawTarget, 10, len(TrainingTarget) + len(ValidationTarget))

# Linear Regression main method.
def LinearRegression(Lamda, LearningRate, Clusters):

    kmeans = KMeans(n_clusters = Clusters, random_state=0).fit(TrainingData)
    MuMatrix = kmeans.cluster_centers_

    BigSigma = GenerateBigSigma(TrainingData, MuMatrix)
    TrainingPhi = GetPhiMatrix(TrainingData, MuMatrix, BigSigma)
    ValidationPhi = GetPhiMatrix(ValidationData, MuMatrix, BigSigma)
    TestingPhi = GetPhiMatrix(TestingData, MuMatrix, BigSigma)
    W = GetWeightsClosedForm(TrainingPhi, TrainingTarget)

    SGD_LinearRegression(W, TrainingPhi, ValidationPhi, TestingPhi, TrainingTarget, ValidationTarget, TestingTarget, Lamda, LearningRate)

# Logistic Regression main method.
def LogisticRegression(Lamda, LearningRate, Clusters):

    kmeans = KMeans(n_clusters = Clusters, random_state=0).fit(TrainingData)
    MuMatrix = kmeans.cluster_centers_

    BigSigma = GenerateBigSigma(TrainingData, MuMatrix)
    TrainingPhi = GetPhiMatrix(TrainingData, MuMatrix, BigSigma)
    ValidationPhi = GetPhiMatrix(ValidationData, MuMatrix, BigSigma)
    TestingPhi = GetPhiMatrix(TestingData, MuMatrix, BigSigma)
    W = GetWeightsClosedForm(TrainingPhi, TrainingTarget)

    SGD_LogisticRegression(W, TrainingPhi, ValidationPhi, TestingPhi, TrainingTarget, ValidationTarget, TestingTarget, Lamda, LearningRate)


# ------------------------- Neural Network Implementation --------------------------- #

# Tensorflow Model

def NeuralNetwork(LearningRate, HiddenLayers, Epochs, BatchSize):

    # Defining Placeholder
    # Placeholders are like variables which are assigned data at a later date.
    # By creating placeholders, we only assign memory(optional) where data is stored later on.
    InputTensor  = tf.placeholder(tf.float32, [None, len(TrainingData[0])])
    OutputTensor = tf.placeholder(tf.float32, [None, 2])

    # Initializing the weights to Normal Distribution
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape,stddev=0.01))

    # Initializing the input to hidden layer weights
    InputLayerWeights  = init_weights([len(TrainingData[0]), HiddenLayers])
    # Initializing the hidden to output layer weights
    OutputLayerWeights = init_weights([HiddenLayers, 2])

    # Computing values at the hidden layer
    # relu to convert to linear data
    HiddenLayer = tf.nn.relu(tf.matmul(InputTensor, InputLayerWeights))
    # Computing values at the output layer
    OutputLayer = tf.matmul(HiddenLayer, OutputLayerWeights)
    
    # Defining Error Function
    # It computes inaccuracy of predictions in classification problems.
    ErrorFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = OutputLayer, labels = OutputTensor))
    # Defining Learning Algorithm and Training Parameters.
    Training = tf.train.GradientDescentOptimizer(LearningRate).minimize(ErrorFunction)

    # Prediction Function.
    Prediction = tf.argmax(OutputLayer, 1)

    # Training the Model.
    TrainingAccuracy = []
    ValidationAccuracy = []

    TrainingData1 = np.array(TrainingData)
    TrainingTarget1 = np_utils.to_categorical(np.array(TrainingTarget),2)
    ValidationData1 = np.array(ValidationData)
    ValidationTarget1 = np_utils.to_categorical(np.array(ValidationTarget),2)
    TestingData1 = np.array(TestingData)

    # TensorFlow session is an object where all operations are run.
    with tf.Session() as Session:

        tf.global_variables_initializer().run()

        # Iterating for the number of Epochs.
        for Epoch in tqdm_notebook(range(Epochs)):

            p = np.random.permutation(range(len(TrainingData)))

            TrainingD = TrainingData1[p]
            TrainingT = TrainingTarget1[p]

            for Start in range(0, len(TrainingD), BatchSize):
                End = Start + BatchSize
                Session.run(Training, feed_dict = { InputTensor: TrainingD[Start:End], 
                                          OutputTensor: TrainingT[Start:End] })

            TrainingAccuracy.append(np.mean(np.argmax(TrainingT, axis=1) ==
                             Session.run(Prediction, feed_dict={InputTensor: TrainingD,
                                                             OutputTensor: TrainingT})))

            ValidationAccuracy.append(np.mean(np.argmax(ValidationTarget1, axis=1) ==
                             Session.run(Prediction, feed_dict={InputTensor: ValidationData1,
                                                             OutputTensor: ValidationTarget1})))

        PredictedTestTarget = Session.run(Prediction, feed_dict={InputTensor: TestingData1})
    
    Right = 0
    Wrong = 0
    for i,j in zip(TestingTarget, PredictedTestTarget):
        
        if np.argmax(i) == j:
            Right = Right + 1
        else:
            Wrong = Wrong + 1

    print ('UBITname      = *****')
    print ('Person Number = *****')
    print ('--------------------------------------------------')
    print ('-------------Neural Network-----------------------')
    print ('--------------------------------------------------')
    print("Errors: " + str(Wrong), " Correct :" + str(Right))
    print("Testing Accuracy: " + str(Right / (Right + Wrong)*100))
    print("Validation Accuracy: " + str(max(ValidationAccuracy)*100))


# Taking the Human Observed concatenation dataset to model linear regression,
# logistic regression and neural network.
print("\n------------Human Observed Dataset - Concatenation--------------")
ReadData('HumanObserved-Features-Data/HumanObservedDataset-Concatenation.csv')
#LinearRegression(0.7, 0.04, 10)
LogisticRegression(0.5, 0.01, 10)
ReadData('HumanObserved-Features-Data/HumanObservedDataset-Concatenation-NN.csv')
NeuralNetwork(0.1, 50, 100, 128)

# Taking the Human Observed subtraction dataset to model linear regression,
# logistic regression and neural network.
print("\n------------Human Observed Dataset - Subtraction--------------")
ReadData('HumanObserved-Features-Data/HumanObservedDataset-Subtraction.csv')
LinearRegression(0.5, 0.05, 10)
LogisticRegression(0.5, 0.01, 10)
ReadData('HumanObserved-Features-Data/HumanObservedDataset-Subtraction-NN.csv')
NeuralNetwork(0.1, 50, 100, 128)

# Taking the GSC concatenation dataset to model linear regression,
# logistic regression and neural network.
print("\n------------GSC Dataset - Concatenation--------------")
print("-------------This may take a while--------------------")
ReadData('GSC-Features-Data/GSC-Dataset-Concatenation.csv')
LinearRegression(1, 0.06, 10)
LogisticRegression(0.5, 0.08, 10)
NeuralNetwork(0.01, 50, 100, 128)

# Taking the GSC subtraction dataset to model linear regression,
# logistic regression and neural network.
print("\n------------GSC Dataset - Subtraction--------------")
print("-------------This may take a while--------------------")
ReadData('GSC-Features-Data/GSC-Dataset-Subtraction.csv')
LinearRegression(0.5, 0.07, 10)
LogisticRegression(0.5, 0.2, 10)
NeuralNetwork(0.01, 50, 100, 128)
