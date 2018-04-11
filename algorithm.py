#import
from random import *
import numpy
import variable


def main():

    create_training_data_set(variable.image_A, variable.Train_data_set_A)
    init_weight()
    print(variable.Weight_input_hidden1)
    print(variable.Weight_hidden1_hidden2)
    print(variable.Weight_hidden2_output)

    for i in range(2):
        for j in range(17):
            print("=============" + str(i) + "----" + str(j) + "=================")
            training_nn(0, variable.Train_data_set_A, j)

    #print(variable.Weight_input_hidden1)
    #print(variable.Weight_hidden1_hidden2)
    #print(variable.Weight_hidden2_output)


def counting(list):
    count = 0
    for i in range(len(list)):
        for j in range(len(list[i])):
            count+=1

    print(count)


def create_training_data_set(image, alphabet_list):
    for i in range(variable.NUM_IMAGE_IN_DATA_SET):
        #print(image)
        alphabet_list.append(scramble(image,variable.RATE))
        #print(alphabet_list)



def scramble(image, rate):
    row = len(image)
    column = len(image[0])
    new_image = []

    for i in range(row):
        new_image.insert(i, [])
        for j in range(column):
            if random() < rate:
                if image[i][j] == 0:
                    new_image[i].insert(j, 1)
                else:
                    new_image[i].insert(j, 0)
            else:
                new_image[i].insert(j, image[i][j])
    #print("new image => %r" % (new_image))

    return new_image


def init_weight():
    #input_hidden1
    for i in range(variable.NUM_NODE_INPUT):
        variable.Weight_input_hidden1.insert(i, [])
        variable.DW_input_hidden1.insert(i, [])
        for j in range(variable.NUM_NODE_HIDDEN1):
            variable.Weight_input_hidden1[i].append(random())
            variable.DW_input_hidden1[i].append(0)
        normalize_weight(variable.Weight_input_hidden1[i])

    #hidden1_hidden2
    for i in range(variable.NUM_NODE_HIDDEN1):
        variable.Weight_hidden1_hidden2.insert(i, [])
        variable.DW_hidden1_hidden2.insert(i, [])
        for j in range(variable.NUM_NODE_HIDDEN2):
            variable.Weight_hidden1_hidden2[i].append(random())
            variable.DW_hidden1_hidden2[i].append(0)
        normalize_weight(variable.Weight_hidden1_hidden2[i])

    #hidden2_output
    for i in range(variable.NUM_NODE_HIDDEN2):
        variable.Weight_hidden2_output.insert(i, [])
        variable.DW_hidden2_output.insert(i, [])
        for j in range(variable.NUM_NODE_OUTPUT):
            variable.Weight_hidden2_output[i].append(random())
            variable.DW_hidden2_output[i].append(0)
        normalize_weight(variable.Weight_hidden2_output[i])


def normalize_weight(weight):
    weight_sum = 0
    for i in range(len(weight)):
        weight_sum += weight[i]

    for i in range(len(weight)):
        weight[i] /= weight_sum


def hyperbolic_tangent(number):
    return ((2 * 1.7) / (1 + numpy.exp(-0.7 * number))) - 1.7
    #return numpy.tanh(number)


def o_input(image):
    variable.Output_input.clear()
    #print(len(variable.Output_input))
    count = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            variable.Output_input.insert(count, image[i][j])
            count += 1

    print("Output input")
    print(variable.Output_input)


def o_hidden1():
    variable.Output_hidden_1.clear()
    #print(len(variable.Output_hidden_1))
    for i in range(variable.NUM_NODE_HIDDEN1):
        sum_output = 0
        for j in range(variable.NUM_NODE_INPUT):
            # print(str(variable.Output_input[j]) + "==========" + str(variable.Weight_input_hidden1[j][i]))
            sum_output += variable.Output_input[j] * variable.Weight_input_hidden1[j][i]
        sum_output -= variable.THRESHOLD
        #print(sum_output)
        activition = hyperbolic_tangent(sum_output)
        variable.Output_hidden_1.insert(i, activition)
        numpy.round(variable.Output_hidden_1, 6)
    print("Output hidden1")
    print(variable.Output_hidden_1)


def o_hidden2():
    variable.Output_hidden_2.clear()
    #print(len(variable.Output_hidden_2))
    for i in range(variable.NUM_NODE_HIDDEN2):
        sum_output = 0
        for j in range(variable.NUM_NODE_HIDDEN1):
            sum_output += variable.Output_hidden_1[j] * variable.Weight_hidden1_hidden2[j][i]
        sum_output -= variable.THRESHOLD
        #print(sum_output)
        activition = hyperbolic_tangent(sum_output)
        variable.Output_hidden_2.insert(i, activition)
        numpy.round(variable.Output_hidden_2, 6)
    print("Output hidden2")
    print(variable.Output_hidden_2)


def o_output():
    variable.Output_output.clear()
    #print(len(variable.Output_output))
    for i in range(variable.NUM_NODE_OUTPUT):
        sum_output = 0
        for j in range(variable.NUM_NODE_HIDDEN2):
            sum_output += variable.Output_hidden_2[j] * variable.Weight_hidden2_output[j][i]
        sum_output -= variable.THRESHOLD
        #print(sum_output)
        activition = hyperbolic_tangent(sum_output)
        variable.Output_output.insert(i, activition)
        numpy.round(variable.Output_output, 6)
    print("Output output")
    print(variable.Output_output)


def error(id):
    variable.Error.clear()
    #print(len(variable.Error))
    for i in range(variable.NUM_NODE_OUTPUT):
        variable.Error.insert(i, variable.expect_output[id][i] - variable.Output_output[i])
    print("Error")
    print(variable.Error)


def d_output():
    variable.Delta_output.clear()
    #print(len(variable.Delta_output))
    for i in range(variable.NUM_NODE_OUTPUT):
        variable.Delta_output.insert(i, variable.Output_output[i]
                                     * (1 - variable.Output_output[i]) * variable.Error[i])
    numpy.round(variable.Delta_output, 6)
    print("Delta output")
    print(variable.Delta_output)


def d_hidden2():
    variable.Delta_hidden2.clear()
    #print(len(variable.Delta_hidden2))
    for i in range(variable.NUM_NODE_HIDDEN2):
        sum_delta = 0
        for j in range(variable.NUM_NODE_OUTPUT):
            sum_delta += variable.Weight_hidden2_output[i][j] * variable.Delta_output[j]
        variable.Delta_hidden2.insert(i, variable.Output_hidden_2[i]
                                          * (1 - variable.Output_hidden_2[i]) * sum_delta)
    numpy.round(variable.Delta_hidden2, 6)
    print("Delta hidden2")
    print(variable.Delta_hidden2)


def d_hidden1():
    variable.Delta_hidden1.clear()
    #print(len(variable.Delta_hidden1))
    for i in range(variable.NUM_NODE_HIDDEN1):
        sum_delta = 0
        for j in range(variable.NUM_NODE_HIDDEN2):
            sum_delta += variable.Weight_hidden1_hidden2[i][j] * variable.Delta_hidden2[j]
        variable.Delta_hidden1.insert(i, variable.Output_hidden_1[i]
                                          * (1 - variable.Output_hidden_1[i]) * sum_delta)
    numpy.round(variable.Delta_hidden1, 6)
    print("Delta hidden1")
    print(variable.Delta_hidden1)


def update_delta_weight_hidden2_output():
    for i in range(variable.NUM_NODE_HIDDEN2):
        for j in range(variable.NUM_NODE_OUTPUT):
            delta_weight = variable.ALPHA * variable.Output_hidden_2[i] * variable.Delta_output[j] \
                           + variable.BETA * variable.DW_hidden2_output[i][j]
            variable.DW_hidden2_output[i][j] = delta_weight
    numpy.round(variable.DW_hidden2_output, 6)
    print("Delta weight hidden2 output")
    print(variable.DW_hidden2_output)


def update_weight_hidden2_output():
    for i in range(variable.NUM_NODE_HIDDEN2):
        for j in range(variable.NUM_NODE_OUTPUT):
            new_weight = variable.Weight_hidden2_output[i][j] + variable.DW_hidden2_output[i][j]
            variable.Weight_hidden2_output[i][j] = new_weight
    numpy.round(variable.Weight_hidden2_output, 6)
    print("New weight hidden2 output")
    print(variable.Weight_hidden2_output)


def update_delta_weight_hidden1_hidden2():
    for i in range(variable.NUM_NODE_HIDDEN1):
        for j in range(variable.NUM_NODE_HIDDEN2):
            delta_weight = variable.ALPHA * variable.Output_hidden_1[i] * variable.Delta_hidden2[j] \
                           + variable.BETA * variable.DW_hidden1_hidden2[i][j]
            variable.DW_hidden1_hidden2[i][j] = delta_weight
    numpy.round(variable.DW_hidden1_hidden2, 6)
    print("Delta Weight hidden1 hidden2")
    print(variable.DW_hidden1_hidden2)


def update_weight_hidden1_hidden2():
    for i in range(variable.NUM_NODE_HIDDEN1):
        for j in range(variable.NUM_NODE_HIDDEN2):
            new_weight = variable.Weight_hidden1_hidden2[i][j] + variable.DW_hidden1_hidden2[i][j]
            variable.Weight_hidden1_hidden2[i][j] = new_weight
    numpy.round(variable.Weight_hidden1_hidden2, 6)
    print("New weight hidden1 hidden2")
    print(variable.Weight_hidden1_hidden2)


def update_delta_weight_input_hidden1():
    for i in range(variable.NUM_NODE_INPUT):
        for j in range(variable.NUM_NODE_HIDDEN1):
            delta_weight = variable.ALPHA * variable.Output_input[i] * variable.Delta_hidden1[j] \
                           + variable.BETA * variable.DW_input_hidden1[i][j]
            variable.DW_input_hidden1[i][j] = delta_weight
    numpy.round(variable.DW_input_hidden1, 6)
    print("Weight Delta input hidden1")
    print(variable.DW_input_hidden1)


def update_weight_input_hidden1():
    for i in range(variable.NUM_NODE_INPUT):
        for j in range(variable.NUM_NODE_HIDDEN1):
            new_weight = variable.Weight_input_hidden1[i][j] + variable.DW_input_hidden1[i][j]
            variable.Weight_input_hidden1[i][j] = new_weight
    numpy.round(variable.Weight_input_hidden1)
    print("New weight input hidden1")
    print(variable.Weight_input_hidden1)


def training_nn(alpha_id, train_data, train_data_id):
    o_input(train_data[train_data_id])
    o_hidden1()
    o_hidden2()
    o_output()

    error(alpha_id)
    d_output()
    d_hidden2()
    d_hidden1()

    update_delta_weight_hidden2_output()
    update_weight_hidden2_output()
    update_delta_weight_hidden1_hidden2()
    update_weight_hidden1_hidden2()
    update_delta_weight_input_hidden1()
    update_weight_input_hidden1()


#main()