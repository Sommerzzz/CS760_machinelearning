Homework Assignment #3
Due at midnight on Friday, 10/27

Part A - Programming

For this part of the assignment, you will be writing code to train and test a neural network with one hidden layer using backpropagation. Specifically, you should assume:
Your code is intended for binary classification problems.
All of the attributes are numeric.
The neural network has connections between input and the hidden layer, and between the hidden and output layer and one bias unit and one output node.
The number of units in the hidden layer should be equal to the number of input units.
For training the neural network, use n-fold stratified cross validation.
Use sigmoid activation function and train using stochastic gradient descent.
Randomly set initial weights for all units including bias in the range (-1,1).
Use a threshold value of 0.5. If the sigmoidal output is less than 0.5, take the prediction to be the class listed first in the ARFF file in the class attributes section; else take the prediction to be the class listed second in the ARFF file.
File Format:

Your program should read files that are in the ARFF format. In this format, each instance is described on a single line. The feature values are separated by commas, and the last value on each line is the class label of the instance. Each ARFF file starts with a header section describing the features and the class labels. Lines starting with '%' are comments. Your program needs to handle only numeric attributes, and simple ARFF files (i.e. don't worry about sparse ARFF files and instance weights). Your program can assume that the class attribute is named 'class' and it is the last attribute listed in the header section. 

Use the following data set for your program : sonar.arff
Specifications:

The program should be callable from command line as follows: 
neuralnet trainfile num_folds learning_rate num_epochs 

Your program should print the output in the following format for each instance in the source file (in the same order in which the instances appear in the source file) 
fold_of_instance predicted_class actual_class confidence_of_prediction 

If you are using a language that is not compiled to machine code (e.g. Java), then you should make a small script called 'neuralnet' that accepts the command-line arguments and invokes the appropriate source-code program and interpreter. More instructions below!
Part B - Analysis

In this section, you will draw graphs for analysing the performance of neural network (using sonar.arff as the data set) with respect to certain parameters.
Plot accuracy of the neural network constructed for 25, 50, 75 and 100 epochs. 
(With learning rate = 0.1 and number of folds = 10)
Plot accuracy of the neural network constructed with number of folds as 5, 10, 15, 20 and 25. 
(With learning rate = 0.1 and number of epochs = 50)
Plot ROC curve for the neural network constructed with the following parameters: 
(With learning rate = 0.1, number of epochs = 50, number of folds = 10)
Please make sure you create three graphs in total (One for each question). Combine all the three graphs in a single PDF file named <wiscid>_analysis.pdf. 
Submission Instructions

Create an executable that calls your program as in Homework Assignment #1. 

Create a directory named <yourwiscID_hw3> . This directory should contain
Your source files in a sub-directory named <src>.
The executable shell script called 'neuralnet'.
The PDF file '<wiscid>_analysis.pdf' containing the graphs.
Jar files or any other artifacts necessary to execute your code.
Compress this directory and submit the compressed zip file in canvas.
Note:

You need to ensure that your code will run, when called from the command line as described above, on the CS department Linux machines.
You WILL be penalized if your program fails to meet any of the above specifications.
Make sure to test your programs on CSL machines before you submit.
