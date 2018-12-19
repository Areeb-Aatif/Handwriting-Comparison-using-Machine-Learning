import numpy as np 
import csv  
import pandas as pd

# This file is used to concatenate and subtract features of same pairs and different pairs of
# Human Observed Dataset and GSC Dataset. 

# Retrieve all pairs from diffn_pairs.csv
def DiffnPairs(filename):
    
    matrix_diffn = []
    with open(filename) as f:
        reader1 = csv.reader(f)
        next(reader1, None)
        for row in reader1:
            l = []
            for column in row[:-1]:
                l.append(column)
            l.append(int(row[-1]))
            matrix_diffn.append(l)

    return matrix_diffn

# Retrieve all pairs from same_pairs.csv
def SamePairs(filename):
    
    matrix_same = []
    with open(filename) as f:
        reader1 = csv.reader(f)
        next(reader1, None)
        for row in reader1:
            l = []
            for column in row[:-1]:
                l.append(column)
            l.append(int(row[-1]))
            matrix_same.append(l)
        
    return matrix_same

# Retrieve all the writers with their features from GSC-Features.csv
def GSCFeaturesMatrix(filename):
    
    matrix = {}
    with open(filename) as f:
        reader1 = csv.reader(f)
        next(reader1, None)
        for row in reader1:
            l = []
            for column in row[1:]:
                l.append(int(column))
            matrix[row[0]] = l

    return matrix

# Retrieve all the writers with their features from HumanObserved-Features-Data.csv
def HumanObservedFeaturesMatrix(filename):
    
    matrix = {}
    with open(filename) as f:
        reader1 = csv.reader(f)
        next(reader1, None)
        for row in reader1:
            l = []
            for column in row[2:]:
                l.append(int(column))
            matrix[row[1]] = l

    return matrix

# Here we concatenate the features for the extracted pairs.
def MatrixConcatenation(matrix_same, matrix_diffn, matrix):

    matrix_concat = []
    #result = []
    #headerlist = CreateHeaderConcat(matrix)
    #result.append(headerlist)

    i=0
    for row in matrix_same:
        print(i)
        i=i+1
        if(i == 25000):
            break
        l1 = []
        l1.append(row[0])
        l1.append(row[1])
        l1 += matrix[row[0]] 
        l1 += matrix[row[1]]
        l1.append(row[2])
        matrix_concat.append(l1)
    i=0
    for row in matrix_diffn:
        print(i)
        i=i+1
        if(i == 25000):
            break
        l = []
        l.append(row[0])
        l.append(row[1])
        l += matrix[row[0]] 
        l += matrix[row[1]]
        l.append(row[2])
        matrix_concat.append(l)
    
    np.random.shuffle(matrix_concat)
    return matrix_concat

# Here we subtract the features for the extracted pairs.
def MatrixSubtraction(matrix_same, matrix_diffn, matrix):

    matrix_sub = []

    #headerlist = CreateHeaderSub(matrix)
    #matrix_sub.append(headerlist)
    i=0
    for row in matrix_same:
        print(i)
        i=i+1
        if(i == 25000):
            break
        l1 = []
        l1.append(row[0])
        l1.append(row[1])
        l2 = matrix[row[0]]
        l3 = matrix[row[1]]
        for col1, col2 in zip(l2, l3):
            l1.append(abs(col1-col2))
        l1.append(row[2])
        matrix_sub.append(l1)

    i = 0
    for row in matrix_diffn:
        print(i)
        i = i+1
        if(i == 25000):
            break
        l = []
        l.append(row[0])
        l.append(row[1])
        l2 = matrix[row[0]]
        l3 = matrix[row[1]]
        for col1, col2 in zip(l2, l3):
            l.append(abs(col1-col2))
        l.append(row[2])
        matrix_sub.append(l)
    
    np.random.shuffle(matrix_sub)
    return matrix_sub

def CreateHeaderConcat(matrix):

    headerlist = ['img_id_A', 'img_id_B']
    l = list(matrix.values())[0]

    for i in range(len(l)):
        headerlist.append('fA'+str(i+1))
    for i in range(len(l)):
        headerlist.append('fB'+str(i+1))
    headerlist.append('t')

    return headerlist

def CreateHeaderSub(matrix):

    headerlist = ['img_id_A', 'img_id_B']
    l = list(matrix.values())[0]

    for i in range(len(l)):
        headerlist.append('|fA'+str(i+1)+' - fB'+str(i+1)+'|')
    headerlist.append('t')

    return headerlist

def CreateCSV(matrix, filename):

    df = pd.DataFrame(np.array(matrix))
    df.to_csv(filename, index = False)

def GSCFeaturesDataset():

    matrix_same = SamePairs('GSC-Features-Data/same_pairs.csv')
    matrix_diffn = DiffnPairs('GSC-Features-Data/diffn_pairs.csv')
    matrix = GSCFeaturesMatrix('GSC-Features-Data/GSC-Features.csv')

    matrix_concat = MatrixConcatenation(matrix_same, matrix_diffn, matrix)
    CreateCSV(matrix_concat, 'GSC-Features-Data/GSC-Dataset-Concatenation-NN.csv')

    matrix_sub = MatrixSubtraction(matrix_same, matrix_diffn, matrix)
    CreateCSV(matrix_sub, 'GSC-Features-Data/GSC-Dataset-Subtraction-NN.csv')

def HumanObservedDataset():

    matrix_same = SamePairs('HumanObserved-Features-Data/same_pairs.csv')
    matrix_diffn = DiffnPairs('HumanObserved-Features-Data/diffn_pairs.csv')
    matrix = HumanObservedFeaturesMatrix('HumanObserved-Features-Data/HumanObserved-Features-Data.csv')

    matrix_concat = MatrixConcatenation(matrix_same, matrix_diffn, matrix)
    CreateCSV(matrix_concat, 'HumanObserved-Features-Data/HumanObservedDataset-Concatenation-NN.csv')

    matrix_sub = MatrixSubtraction(matrix_same, matrix_diffn, matrix)
    np.random.shuffle(matrix_sub)
    CreateCSV(matrix_sub, 'HumanObserved-Features-Data/HumanObservedDataset-Subtraction-NN.csv')

GSCFeaturesDataset()

# HumanObservedDataset()

