''' 

                                            Neural Network.py
                                                2022-12-1
                                            written by  Zhang Hui

'''

import numpy
# scipy.special for the sigmiod function expit()
import scipy.special
import matplotlib.pyplot

# Neural network class definition
class NeuralNetwork:

    # initialise the neural network
    def __init__( self , inputnodes , hiddennodes , outputnodes , learningrate ):

        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # link weight matrices, wih and who
        self.wih = (numpy.random.rand(self.hnodes, self.inodes)-0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes)-0.5)
        self.wih_1 = numpy.random.normal(0.0 , pow( self.hnodes , -0.5) , (self.hnodes,self.inodes))
        self.who_1 = numpy.random.normal(0.0 , pow( self.onodes , -0.5) , (self.onodes,self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = lambda x : scipy.special.expit(x)

        pass

    # train the neural network
    def __train__( self , inputs_list , targets_list):

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T , output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), numpy.transpose(inputs))

        print(" Testing work has been finished!")
        pass

    # query the neural network
    def __query__( self , inputs_list ):

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list , ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #print
        print(" Testing work has been finished!")
        return final_outputs

'''---------------------------------------------------------------------------------------------------------------------'''
print(" Neural Network is working  ...")
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.1
lesrning_rate = 0.1

# create instance of neural network
NN = NeuralNetwork( input_nodes , hidden_nodes , output_nodes , lesrning_rate )
print(" Neural Network has been created ...")

# load the mnist training data CSV file into a list
print(" Load the mnist training data CSV file into a list ...")
Training_data_file = open("mnist_train.csv", 'r')
Training_data_list = Training_data_file.readlines()
Training_data_file.close()
print(" Load-working has been finished!")

# train the neural network

# epochs is the number of times the training data set is used for training
print("Begin Epoch ...")
epochs = 5
num = 1
for e in range(epochs):
    # go through all records in the training data set
    for record in Training_data_list:
        all_values = record.split(',')
        image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        
        matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
        matplotlib.pyplot.ion()
        matplotlib.pyplot.pause(0.0001)
        matplotlib.pyplot.close()
        # scale and shift the inputs

        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        NN.__train__(inputs, targets)
        print("Target is", int(all_values[0]))
        print("The ", num , "-th training work has been finished!")
        num = num + 1
        pass
    pass

# load the mnist test data CSV file into a list

print(" Load the mnist test data CSV file into a list ...")
Test_data_file = open("mnist_test.csv", 'r')
Test_data_list = Test_data_file.readlines()
Test_data_file.close()
print(" Load-working has been finished!")

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []
num = 1
# go through all the records in the test data set
for record in Test_data_list:
    all_values = record.split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.ion()
    matplotlib.pyplot.pause(0.1)
    matplotlib.pyplot.close()
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = NN.__query__(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("correct_label is", correct_label, ",and"," The result of Neural Network is ", label)
    print("The ", num, "-th testing work has been finished!")
    num = num + 1
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
        print("Successful!")

    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        print("Fualt!")
        pass
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum()/scorecard_array.size)
