Homework Assignment #4
Due Friday, 11/10

Part 1

For this homework, you are to write a program that implements both naive Bayes and TAN (tree-augmented naive Bayes). Specifically, you should assume:
Your code is intended for binary classification problems.
All of the variables are discrete valued. Your program should be able to handle an arbitrary number of variables with possibly different numbers of values for each variable.
Laplace estimates (pseudocounts of 1) are used when estimating all probabilities.
For the TAN algorithm. Your program should:

Use Prim's algorithm to find a maximal spanning tree (but choose maximal weight edges instead of minimal weight ones). Initialize this process by choosing the first variable in the input file for Vnew. If there are ties in selecting maximum weight edges, use the following preference criteria: (1) prefer edges emanating from variables listed earlier in the input file, (2) if there are multiple maximum weight edges emanating from the first such variable, prefer edges going to variables listed earlier in the input file.
To root the maximal weight spanning tree, pick the first variable in the input file as the root.
Your program should read files that are in the ARFF format. In this format, each instance is described on a single line. The variable values are separated by commas, and the last value on each line is the class label of the instance. Each ARFF file starts with a header section describing the variables and the class labels. Lines starting with '%' are comments. See the link above for a brief, but more detailed description of the ARFF format. Your program needs to handle only discrete variables, and simple ARFF files (i.e. don't worry about sparse ARFF files and instance weights). Example ARFF files are provided below. Your program can assume that the class variable is named 'class' and it is the last variable listed in the header section.

The program should be called bayes and should accept four command-line arguments as follows:
bayes <train-set-file> <test-set-file> <n|t>
where the last argument is a single character (either 'n' or 't') that indicates whether to use naive Bayes or TAN.

If you are using a language that is not compiled to machine code (e.g. Java), then you should make a small script called bayes that accepts the command-line arguments and invokes the appropriate source-code program and interpreter, as you did with the previous homeworks.

Your program should determine the network structure (in the case of TAN) and estimate the model parameters using the given training set, and then classify the instances in the test set. Your program should output the following:

The structure of the Bayes net by listing one line per variable in which you indicate (i) the name of the variable, (ii) the names of its parents in the Bayes net (for naive Bayes, this will simply be the 'class' variable for each other variable) separated by whitespace.
One line for each instance in the test-set (in the same order as this file) indicating (i) the predicted class, (ii) the actual class, (iii) and the posterior probability of the predicted class (rounded to 12 digits after the decimal point).
The number of the test-set examples that were correctly classified.
You can test the correctness of your code using lymph_train.arff and lymph_test.arff, as well as vote_train.arff and vote_test.arff. This directory contains the outputs your code should produce for each data set.

Part 2

For this part, use stratified 10-fold cross validation on the chess-KingRookVKingPawn.arff data set to compare naive Bayes and TAN. Be sure to use the same partitioning of the data set for both algorithms. Report the accuracy the models achieve for each fold and then use a paired t-test to determine the statistical significance of the difference in accuracy. Report both the value of the t-statistic and the resulting p value.
You can use a t-test calculator, such as this one for this exercise.

Submitting Your Work

You should turn in your work electronically using the Canvas course management system. Turn in all source files and your runnable program as well as a file called hw4.pdf that shows your work for Part 2. All files should be compressed as one zip file named <Wisc username>_hw4.zip. Upload this zip file at the course Canvas site.
