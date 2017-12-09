import sys
import arff
from operator import itemgetter
from math import log


class naiveBayes(object):

    def __init__(self, data, labels, features):
        '''
        initialize variables
        '''
        self.data = data
        self.labels = labels
        self.features = features
        self.model()
        
    def model(self):
        '''
        build model
        '''
        self.length = len(self.data)
        self.outcomes = []
        self.covariates = []
        self.labelCount = {} # p(y)
        self.featureCount = {} # p(x_i|y)
        
        # initialize featureCount
        for featureIndex, (featureName, featureValue) in enumerate(self.features):
            
            self.featureCount[featureIndex] = {}
            
            for label in self.labels:
                self.featureCount[featureIndex][label] = {}
                for value in featureValue:
                    self.featureCount[featureIndex][label][value] = 1
        
        # initialize labelCount             
        for label in self.labels:
            self.labelCount[label] = 1
        
        # calculate D, D_y, D_{y,x_i}    
        for instance in self.data:
            
            outcome = instance[-1]
            covariate = instance[:-1]
            
            self.outcomes.append(outcome)
            self.covariates.append(covariate)
            self.labelCount[outcome] += 1
            
            for index, cov in enumerate(covariate):
                self.featureCount[index][outcome][cov] += 1
        
        # calculate p(y) -> labelCount, p(x_i|y) -> featureCount        
        for featureIndex, (featureName, featureValue) in enumerate(self.features):
            for label in self.labels:
                count = sum(self.featureCount[featureIndex][label].values())
                for value in featureValue:
                    self.featureCount[featureIndex][label][value] *= 1.0/count
        labelLength = sum(self.labelCount.values())
        self.labelCount = {key: float(val)/labelLength for key,val in self.labelCount.items()}
        
    def predict(self, dataTest, pr):
        '''
        predict dataTest (pr: print or not)
        '''
        if pr == True:
            for (featureName, featureValue) in self.features:
                print featureName, 'class'
            print ''
            
        right = 0
        for instance in dataTest:
            
            outcome = instance[-1]
            covariate = instance[:-1]
            prob = {}
            
            # h_nb = argmax p(y)p(x_1|y)...p(x_d|y)
            for label in self.labels:
                prob[label] = self.labelCount[label]
                for index in range(len(covariate)):
                    prob[label] *= self.featureCount[index][label][covariate[index]]
                    
            probC = sum(prob.values())
            
            for key, val in prob.items():
                prob[key] = float(val)/probC
                
            predictClass = max(prob.iteritems(), key = itemgetter(1))
            if pr == True:
                print "{} {} {:.12f}".format(predictClass[0], outcome, predictClass[1])
            if predictClass[0] == outcome:
                right += 1
        if pr == True:
            print '\n', right
        self.accuracy = float(right)/len(dataTest)

class TAN(object):

    def __init__(self, data, labels, features):
        '''
        initialize variables
        '''
        self.data = data
        self.labels = labels
        self.features = features
        self.weights()
        
    def weights(self):
        '''
        build model
        '''
        self.length = len(self.data)
        self.outcomes = []
        self.covariates = []
        self.labelCount = {} # p(y)
        self.featureCount = {} # p(x_i|y)
        self.jointCount = {} # p(x_i,x_j|y)
        self.information = {} # I(x_i,x_j|y)
        self.condition = {} # p(x_j|x_i,y)
        
        # initialize featureCount & jointCount
        featureLength = len(self.features)
        for featureIndex, (featureName,featureValue) in enumerate(self.features):
            self.featureCount[featureIndex] = {}
            self.jointCount[featureIndex] = {}
            for label in self.labels:
                self.featureCount[featureIndex][label] = {}
                for value in featureValue:
                    self.featureCount[featureIndex][label][value] = 1
            for jointIndex, (jointName, jointValue) in enumerate(self.features):
                self.jointCount[(featureIndex, jointIndex)] = {}
                for label in self.labels:
                    self.jointCount[(featureIndex, jointIndex)][label] = {}
                    for fvalue in featureValue:
                        for jvalue in jointValue:
                            self.jointCount[(featureIndex, jointIndex)][label][(fvalue, jvalue)] = 1
                            
        # initialize labelCount
        for label in self.labels:
            self.labelCount[label] = 1
            
        # calculate featureCount & jointCount
        for dataIndex, instance in enumerate(self.data):
            
            outcome = instance[-1]
            covariate = instance[:-1]
            
            self.outcomes.append(outcome)
            self.covariates.append(covariate)
            self.labelCount[outcome] += 1
            
            for iIndex, iCovariate in enumerate(covariate):
                self.featureCount[iIndex][outcome][iCovariate] += 1
                for jIndex, jCovariate in enumerate(covariate):
                    self.jointCount[(iIndex, jIndex)][outcome][(iCovariate, jCovariate)] += 1
        
        # calculate p(y) -> labelCount, p(x_i|y) -> featureCount        
        for featureIndex, (featureName, featureValue) in enumerate(self.features):
            for label in self.labels:
                count = sum(self.featureCount[featureIndex][label].values())
                for value in featureValue:
                    self.featureCount[featureIndex][label][value] *= 1.0/count
        
        # calculate I(x_i,x_j|y) -> information
        for iIndex, (iName, iValue) in enumerate(self.features):
            for jIndex, (jName, jValue) in enumerate(self.features):
                self.information[(iIndex, jIndex)] = 0
                for label in self.labels:
                    for (key1, key2), val in self.jointCount[(iIndex, jIndex)][label].items():
                        p1 = float(val)/(self.length + 2*len(iValue)*len(jValue))
                        p2 = float(val)/(self.labelCount[label] - 1 + len(iValue)*len(jValue))
                        if iIndex == jIndex:
                            self.information[(iIndex, jIndex)] = -1
                            break
                        else:
                            self.information[(iIndex, jIndex)] += (p1*log(p2/(self.featureCount[iIndex][label][key1]*self.featureCount[jIndex][label][key2]), 2))
        
        # calculate p(x_j|x_i, y) -> condition
        for iIndex, (iName, iValue) in enumerate(self.features):
            for jIndex, (jName, jValue) in enumerate(self.features):
                self.condition[(iIndex, jIndex)] = {}
                for label in self.labels:
                    self.condition[(iIndex, jIndex)][label] = {}
                    for i in iValue:
                        count1 = 0
                        for j in jValue:
                            count1 += self.jointCount[(iIndex, jIndex)][label][(i,j)]
                        for j in jValue:
                            self.condition[(iIndex, jIndex)][label][((i,j))] = self.jointCount[(iIndex, jIndex)][label][(i,j)]*(1.0/count1)
        
    def prim(self):
        '''
        prim's algorithm
        '''
        featureLength = len(self.features)
        vSet = [i for i in range(1, featureLength)]
        vNew = [0]
        vRoot = {}
        self.eNew = {}
        while vSet != []:
            base = 0
            outVec = featureLength + 1
            inVec = featureLength + 1
            for u in vNew:
                for v in vSet:
                    if self.information[(u,v)] > base:
                        base = self.information[(u,v)]
                        outVec = u
                        inVec = v
                    if self.information[(u,v)] == base:
                        if (u <= outVec) and (v <= inVec):
                            outVec = u
                            inVec = v
            vSet.remove(inVec)
            vNew.append(inVec)
            vRoot[inVec] = outVec
            self.eNew[(outVec, inVec)] = base
            pass
        self.vNew = vNew
        self.vRoot = vRoot
        
    def predict(self, dataTest, pr):
        if pr == True:
            for featureIndex, (featureName, featureValue) in enumerate(self.features):
                if featureIndex == 0:
                    print featureName, 'class'
                else:
                    print featureName, self.features[self.vRoot[featureIndex]][0], 'class'
            print ''
        right = 0
        for instance in dataTest:
               
            outcome = instance[-1]
            covariate = instance[:-1]
            prob = {}
        
            for label in self.labels:
                prob[label] = float(self.labelCount[label])/(self.length + 2)
                prob[label] *= self.featureCount[0][label][covariate[0]]
                for (key1, key2) in self.eNew.keys():
                    prob[label] *= self.condition[(key1, key2)][label][(covariate[key1], covariate[key2])]
                    
            probC = sum(prob.values())
            for key, val in prob.items():
                prob[key] = float(val)/probC
            predictClass = max(prob.iteritems(), key = itemgetter(1))
            if pr == True:
                print "{} {} {:.12f}".format(predictClass[0], outcome, predictClass[1])
            if predictClass[0] == outcome:
                right += 1
        
        if pr == True:
            print '\n', right
        self.accuracy = float(right)/len(dataTest)
        
def main():
    arg = sys.argv

    trainData = arff.load(open(arg[1], 'r'))
    testData = arff.load(open(arg[2], 'r'))

    attributes = trainData['attributes']
    features = attributes[:-1]
    labels = attributes[-1][1]

    dataTrain = trainData['data']
    dataTest = testData['data']

    method = arg[3]

    if method == 'n':
	    x = naiveBayes(dataTrain, labels, features)
	    x.predict(dataTest, pr = True)
    elif method == 't':
	    x = TAN(dataTrain, labels, features)
	    x.prim()
	    x.predict(dataTest, pr = True)

if __name__ == '__main__':
    main()
    




