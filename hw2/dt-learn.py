import arff
import tree
import sys

arg = sys.argv
m = int(arg[3])
trainData = arff.load(open(arg[1], 'r'))
testData = arff.load(open(arg[2], 'r'))

myTree = tree.createTree(trainData['data'], trainData['attributes'], m)
tree.plotTree(myTree, trainData['attributes'])

prediction = [tree.classify(myTree, testData['attributes'], obs) for obs in testData['data']]
true = [obs[-1] for obs in testData['data']]
print "<Predictions for the Test Set Instances>"
n = 0
for i in range(len(prediction)):
    index = i + 1   
    if prediction[i] == true[i]:
        n += 1
    print "{}: Actual: {} Predicted: {}".format(n, true[i], prediction[i])
print "Number of correctly classified: {} Total number of test instances: {}".format(n, len(testData['data']))
