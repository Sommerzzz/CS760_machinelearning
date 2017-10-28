import sys
import numpy as np
import random
import arff

def sigmoid(x):
    '''sigmoid function'''
    return 1.0/(1.0 + np.exp(-x))
    
def derivatives_sigmoid(x):
    '''derivative of sigmoid function'''
    return x * (1.0 - x)
    
def encoding(class_list, levels):
    '''encode instance class to 0 and 1'''
    class_list_encoding = np.copy(class_list)
    val0 = levels[1]
    val1 = levels[0]
    for i in range(len(class_list_encoding)):
        if class_list_encoding[i][0] == val0:
            class_list_encoding[i][0] = 0
        elif class_list_encoding[i][0] == val1:
            class_list_encoding[i][0] = 1
    class_list_encoding = class_list_encoding.astype(int)
    return class_list_encoding
    
def SGD(X, y, learning_rate, num_epochs, levels):
    '''stochastic gradient descent to update weights and bias'''
    # variable initialization
    inputlayer_neurons = X.shape[1] # number of features in data set
    hiddenlayer_neurons = inputlayer_neurons # number of hidden layers neurons
    output_neurons = 1 # number of neurons at output layer
    y_encoding = encoding(y, levels)
    
    # weight and bias initialization
    w = np.random.uniform(-1,1,size = (inputlayer_neurons, hiddenlayer_neurons)) # weight matrix to the hidden layer
    b = np.random.uniform(-1,1,size = (1, hiddenlayer_neurons)) # bias matrix to the hidden layer
    u = np.random.uniform(-1,1,size = (hiddenlayer_neurons, output_neurons)) # weight matrix to the output layer
    c = np.random.uniform(-1,1,size = (1, output_neurons))
    
    for i in range(num_epochs):
        # stochastic gradient descent
        data_index = range(X.shape[0])
        random.shuffle(data_index)
        for j in data_index:
            rand_index = j
        
            # forward propagation
            hidden_layer_input1 = np.dot(X[rand_index],w)
            hidden_layer_input = hidden_layer_input1 + b
            hiddenlayer_activations = sigmoid(hidden_layer_input)
            output_layer_input1 = np.dot(hiddenlayer_activations,u)
            output_layer_input = output_layer_input1 + c
            output = sigmoid(output_layer_input)
        
            # backward propagation
            error = y_encoding[rand_index] - output
            slope_output_layer = derivatives_sigmoid(output)
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
            delta_output = error * slope_output_layer
            error_at_hidden_layer = delta_output.dot(u.T)
            delta_hiddenlayer = error_at_hidden_layer * slope_hidden_layer
            u += learning_rate * hiddenlayer_activations.T.dot(delta_output)
            c += learning_rate * delta_output
            w += learning_rate * X[rand_index].reshape(1,X.shape[1]).T.dot(delta_hiddenlayer)
            b += learning_rate * delta_hiddenlayer
        
    return u,c,w,b
    
def predict_class(X_train, y_train, X_test, y_test, learning_rate, num_epochs, levels):
    '''predict class for test set based on trained weights'''
    u,c,w,b = SGD(X_train, y_train, learning_rate, num_epochs, levels)
    
    h = sigmoid(np.dot(X_test, w) + b)
    confidence_of_prediction = sigmoid(np.dot(h, u) + c)
    
    predicted_class = np.array([[0] for i in range(len(confidence_of_prediction))])
    threshold = 0.5
    
    for i in range(len(confidence_of_prediction)):
        if confidence_of_prediction[i] > threshold:
            predicted_class[i] = 1
            
    actual_class = encoding(y_test, levels)
    return predicted_class, actual_class, confidence_of_prediction
    
def cross_validation(X, y, num_folds, learning_rate, num_epochs, levels):
    '''perform n-fold stratified cross validation'''
    y_encoding = encoding(y, levels)
    index = np.array([[i] for i in range(len(X))])
    index_X_y = np.concatenate((index, X, y_encoding), axis = 1)
    
    # stratified n-folds
    positive_instances = index_X_y[index_X_y[:, -1] == 1]
    positive_cvgroup = np.array([[int(random.randint(0, num_folds - 1))] for i in range(len(y[y_encoding == 1]))])
    positive = np.concatenate((positive_cvgroup, positive_instances), axis = 1)
    negative_instances = index_X_y[index_X_y[:, -1] == 0]
    negative_cvgroup = np.array([[int(random.randint(0, num_folds - 1))] for i in range(len(y[y_encoding == 0]))])
    negative = np.concatenate((negative_cvgroup, negative_instances), axis = 1)
    combined = np.concatenate((positive,negative), axis = 0)
    combined = np.array(sorted(combined, key = lambda x : x[1])) # to raw order
    
    # cross validation
    total_print_predicted_class = np.array([]); total_print_predicted_class = total_print_predicted_class.reshape(0,3)
    total_print_actual_class = np.array([]); total_print_actual_class = total_print_actual_class.reshape(0,3)
    total_print_confidence_of_prediction = np.array([]); total_print_confidence_of_prediction = total_print_confidence_of_prediction.reshape(0,3)
    for i in np.linspace(0, num_folds - 1, num = num_folds):
        training_groups = combined[combined[:, 0] != i]
        testing_group = combined[combined[:, 0] == i]
        X_train = training_groups[:, 2:-1]
        y_train = np.array([[int(training_groups[j, -1])] for j in range(len(training_groups))])
        X_test = testing_group[:, 2:-1]
        y_test = np.array([[int(testing_group[j, -1])] for j in range(len(testing_group))])
        predicted_class, actual_class, confidence_of_prediction = predict_class(X_train, y_train, X_test, y_test, learning_rate, num_epochs, levels)
        
        group_num = np.array([i for k in range(len(predicted_class))]).reshape(len(predicted_class),1)
        
        print_predicted_class = np.array([instance[0] for instance in predicted_class])
        print_predicted_class = np.concatenate((group_num, print_predicted_class.reshape(len(print_predicted_class),1), testing_group[:,1].reshape(len(print_predicted_class),1)), axis = 1)
        
        print_actual_class = np.array([instance[0] for instance in actual_class])
        print_actual_class = np.concatenate((group_num, print_actual_class.reshape(len(print_actual_class),1), testing_group[:,1].reshape(len(print_actual_class),1)), axis = 1)
        
        print_confidence_of_prediction = np.array([instance[0] for instance in confidence_of_prediction])
        print_confidence_of_prediction = np.concatenate((group_num, print_confidence_of_prediction.reshape(len(print_confidence_of_prediction),1), testing_group[:,1].reshape(len(print_confidence_of_prediction),1)), axis = 1)

        total_print_predicted_class = np.concatenate((total_print_predicted_class,print_predicted_class), axis = 0)
        total_print_actual_class = np.concatenate((total_print_actual_class,print_actual_class), axis = 0)
        total_print_confidence_of_prediction = np.concatenate((total_print_confidence_of_prediction,print_confidence_of_prediction), axis = 0)
    
    total_print_predicted_class = np.array(sorted(total_print_predicted_class, key = lambda x : x[2]))
    total_print_actual_class = np.array(sorted(total_print_actual_class, key = lambda x : x[2]))
    total_print_confidence_of_prediction = np.array(sorted(total_print_confidence_of_prediction, key = lambda x : x[2]))
    
    group = [j[0] for j in total_print_predicted_class]
    print_predicted_class = [j[1] for j in total_print_predicted_class]
    print_actual_class = [j[1] for j in total_print_actual_class]
    print_confidence_of_prediction = [j[1] for j in total_print_confidence_of_prediction]
    
    print_predicted_y = []; print_actual_y = []
    for n in range(len(print_predicted_class)):
        if print_predicted_class[n] == 0:
            print_predicted_y.append(np.unique(y)[0])
        else:
            print_predicted_y.append(np.unique(y)[1])
    for n in range(len(print_actual_class)):
        if print_actual_class[n] == 0:
            print_actual_y.append(np.unique(y)[0])
        else:
            print_actual_y.append(np.unique(y)[1])
    
    for m in range(len(group)):
        print("%d\t%s\t%s\t%f" %(group[m], print_predicted_y[m], print_actual_y[m], print_confidence_of_prediction[m])) 
    accuracy = 0
    count = 0
    for k in range(len(group)):
        if print_predicted_class[k] == print_actual_class[k]:
            count += 1
    accuracy = float(count)/len(group)
    return len(group), count, accuracy, group, print_predicted_class, print_actual_class, print_confidence_of_prediction

def main(trainfile, num_folds, learning_rate, num_epochs):
    X = np.array(arff.load(open(trainfile, 'r'))['data'])[:,:-1] 
    X = np.array([map(float,X[i]) for i in range(len(X))])
    y = np.array(arff.load(open(trainfile, 'r'))['data'])[:,-1]
    y = np.array([[str(y[i])] for i in range(len(y))])
    levels = np.array(arff.load(open(trainfile, 'r'))['attributes'][-1][1])
    cross_validation(X, y, num_folds, learning_rate, num_epochs, levels)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
