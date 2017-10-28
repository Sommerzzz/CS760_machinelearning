from math import log
import operator

def entropy(dataSet): # data is a list of lists in the form of arff 'data'
    '''calculate entropy for a given data set'''
    labelCounts = {}
    for obs in dataSet:
        obsClass = obs[-1]
        if obsClass in labelCounts.keys():
            labelCounts[obsClass] += 1
        else:
            labelCounts[obsClass] = 0
            labelCounts[obsClass] += 1
    Entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/len(dataSet)
        Entropy -= prob * log(prob,2)
    return Entropy

def splitNominal(dataSet, feature, *value): # feature is the index of nominal candidate split, *value are its values 
    '''split dataset on nomial split'''
    subDataSet = []
    for obs in dataSet:
        if obs[feature] in value:
            reducedObs = obs[:feature]
            reducedObs.extend(obs[feature+1:])
            subDataSet.append(reducedObs)
    return subDataSet

def majorityCount(classList, attributes): # classList is a list of class of each obs, attributes is arff['attributes']
    '''count the frequency of occurence of each class label and get the majority'''
    labelList = attributes[-1][1] # ['positive', 'negative']
    classCount = {}
    for label in labelList:
        classCount[label] = 0
    for var in classList:
        classCount[var] += 1
    sortedClassCount_desc = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    majorClass = sortedClassCount_desc[0][0]
    sortedClassCount_asc = sorted(classCount.iteritems(), key = operator.itemgetter(0))
    classCount = [item[1] for item in sortedClassCount_asc]
    return classCount, majorClass # classCount is number of each class sorted ascending, majorClass is the most frequent class

def thresholdCandidate(dataSet, feature, attributes): 
    '''get a list of threshold candidates for a numeric variable'''
    Candidates = []
    rawValue = [[obs[feature], obs[-1]] for obs in dataSet]
    sortValue = sorted(rawValue, key = operator.itemgetter(0))
    sortValueNum = [val[0] for val in sortValue]
    uniqueValue = sorted(list(set(sortValueNum)))
    for i in range(len(uniqueValue) - 1): # check whether there are two classes between two values
        splitData = splitNominal(dataSet, feature, uniqueValue[i],uniqueValue[i+1])
        classList = [obs[-1] for obs in splitData]
        classCount, majorClass = majorityCount(classList, attributes)
        if 0 in classCount:
            pass
        else:
            Candidates.append((uniqueValue[i]+uniqueValue[i+1])/2.0)           
    return Candidates

def getBestThreshold(dataSet, feature, Candidates):
    '''choose a best threshold to split on a numeric feature from cadidates' list w.r.t information gain'''
    oriEnt = entropy(dataSet); cBestInfoGain = 0.0; bestThreshold = 0.0
    for candidate in Candidates: # split the dataset on each candidate
        leftSubset = []; rightSubset = []
        for obs in dataSet:
            if obs[feature] <= candidate:
                leftSubset.append(obs)
            else:
                rightSubset.append(obs)
        infoGain = oriEnt - (float(len(leftSubset))/(len(dataSet)))*entropy(leftSubset) - (float(len(rightSubset))/(len(dataSet)))*entropy(rightSubset)
        if infoGain > cBestInfoGain:
            cBestInfoGain = infoGain
            bestThreshold = candidate
    return bestThreshold

def splitNumeric(dataSet, feature, threshold):
    '''split dataset on numeric split'''
    leftSubset = []; rightSubset = []
    for obs in dataSet:
        if obs[feature] <= threshold:
            leftSubset.append(obs)
        else:
            rightSubset.append(obs)
    return leftSubset, rightSubset

def bestSplit(dataSet, attributes):
    '''choose the best split (if numeric, give threshold)'''
    oriEnt = entropy(dataSet); bestInfoGain = 0.0
    bestFeat = -1; bestThreshold = 0.0
    for i in range(len(dataSet[0]) - 1):
        if type(dataSet[0][i]) == unicode: # for nominal variables
            Splits = [obs[i] for obs in dataSet]
            uniqueValues = attributes[i][1]
            newEnt = 0.0
            for value in uniqueValues:
                subDataSet = splitNominal(dataSet, i, value)
                prob = float(len(subDataSet))/len(dataSet)
                newEnt += prob*entropy(subDataSet)
        elif type(dataSet[0][i]) == int or type(dataSet[0][i]) == float: # for numeric variables
            Candidates = thresholdCandidate(dataSet, i, attributes)
            threshold = getBestThreshold(dataSet, i, Candidates)
            leftSubset, rightSubset = splitNumeric(dataSet, i, threshold)
            newEnt = (float(len(leftSubset))/(len(dataSet)))*entropy(leftSubset) + (float(len(rightSubset))/(len(dataSet)))*entropy(rightSubset)
        infoGain = oriEnt - newEnt
        if infoGain > bestInfoGain: # get the best information gain
            bestInfoGain = infoGain
            bestFeat = i
            if (type(dataSet[0][i]) == int or type(dataSet[0][i]) == float):
                bestThreshold = threshold
    return bestFeat, bestThreshold

def createTree(dataSet, attributes, m, Class = 1, level = 0):
    '''create the tree'''
    classList = [obs[-1] for obs in dataSet]
    classCount, majorClass = majorityCount(classList, attributes)
    if level > 0: # break tie
        if classCount[0] == classCount[1]:
            majorClass = Class
    if len(dataSet) < m: # there are fewer than m training instances reaching the node, where m is provided as input to the program
        return majorClass
    if classList.count(classList[0]) == len(classList): # all of the training instances reaching the node belong to the same class
        return classList[0]
    if len(dataSet[0]) == 1: # there are no more remaining candidate splits at the node.
        return majorClass
    bestFeat, bestThreshold = bestSplit(dataSet, attributes)
    if type(dataSet[0][bestFeat]) == unicode: # for nominal split
        bestFeatLabel = attributes[bestFeat][0]
        myTree = {bestFeatLabel:{}}
        uniqueVals = attributes[bestFeat][1]
        newAttributes = attributes[:]
        del(newAttributes[bestFeat])
        featValues = [obs[bestFeat] for obs in dataSet]
        for value in uniqueVals:
            subData = splitNominal(dataSet, bestFeat, value)
            classList = [obs[-1] for obs in subData]
            classCount, subMajorClass = majorityCount(classList, attributes)
            classCount = '[{0} {1}]'.format(classCount[0],classCount[1])
            myTree[bestFeatLabel][bestFeatLabel+' = '+value +' '+ classCount] = createTree(subData, newAttributes, m, majorClass, level+1)
    elif type(dataSet[0][bestFeat]) == int or type(dataSet[0][bestFeat]) == float: # for numeric split
        bestFeatLabel = attributes[bestFeat][0]
        myTree = {bestFeatLabel:{}}
        newAttributes = attributes[:]
        leftSubset, rightSubset = splitNumeric(dataSet, bestFeat, bestThreshold)
        newAttributes = attributes[:]
        classListLeft = [obs[-1] for obs in leftSubset]
        classCountLeft, leftMajorClass = majorityCount(classListLeft, attributes)
        classCountLeft = '[{0} {1}]'.format(classCountLeft[0], classCountLeft[1])
        classListRight = [obs[-1] for obs in rightSubset]
        classCountRight, rightMajorClass = majorityCount(classListRight, attributes)
        classCountRight = '[{0} {1}]'.format(classCountRight[0], classCountRight[1])
        myTree[bestFeatLabel][bestFeatLabel+' <= '+ str(format(bestThreshold, '.6f'))+' '+classCountLeft] = createTree(leftSubset, newAttributes, m, majorClass, level+1)
        myTree[bestFeatLabel][bestFeatLabel+' > '+ str(format(bestThreshold, '.6f'))+' '+classCountRight] = createTree(rightSubset, newAttributes, m, majorClass, level+1)
    return myTree

def classify(inputTree, attributes, testVec):
    '''use the tree for classification'''
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featLabels = [obs[0] for obs in attributes]
    featIndex = featLabels.index(firstStr) # e.g. the first split 'thal' indexed by 12 in attributes
    if attributes[featIndex][1] in ['REAL', 'NUMERIC']:
        key = secondDict.keys()
        number = str(key[0]).split()[1:3]
        if eval(str(testVec[featIndex]) + " ".join(number)):
            if type(secondDict[key[0]]).__name__ == 'dict':
                classLabel = classify(secondDict[key[0]], attributes, testVec)
            else:
                classLabel = secondDict[key[0]]
        else:
            if type(secondDict[key[1]]).__name__ == 'dict':
                classLabel = classify(secondDict[key[1]], attributes, testVec)
            else:
                classLabel = secondDict[key[1]]
    else:
        for key in secondDict.keys():
            if testVec[featIndex] == key.split()[2]:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = classify(secondDict[key], attributes, testVec)
                else:
                    classLabel = secondDict[key]
    return classLabel

def plotTree(myTree, attributes, n = 0):
    '''plot the tree'''
    attributes = dict(attributes)
    keys = myTree.keys()
    if len(keys) == 1:
        plotTree(myTree[keys[0]], attributes, n)
    else:
        index = keys[0].split()[0]
        labels = attributes[index]
        if labels in ['REAL', 'NUMERIC']:
            for key in sorted(keys):
                if isinstance(myTree[key], dict):
                    print '{0}{1}'.format("|\t" * n, key)
                    plotTree(myTree[key], attributes, n+1)
                else:
                    print '{0}{1}{2}'.format("|\t" * n, key, str(': ') + myTree[key])
        else:
            rightKeys = []
            for i in range(len(labels)):
                for j in range(len(keys)):
                    if labels[i] == keys[j].split()[2]:
                        rightKeys.append(keys[j])
            for key in rightKeys:
                if (isinstance(myTree[key], dict)):
                    print '{0}{1}'.format("|\t" * n,key)
                    plotTree(myTree[key], attributes, n+1)
                else: 
                    print '{0}{1}{2}'.format("|\t" * n, key, str(': ') + myTree[key])
