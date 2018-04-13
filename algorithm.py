from random import *
import numpy
import variable


def create_training_data_set(image, alphabet_list):
    for i in range(variable.NUM_IMAGE_IN_DATA_SET):
        alphabet_list.append(scramble(image,variable.RATE))


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


def o_input(image):
    variable.Output_input.clear()
    count = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            variable.Output_input.insert(count, image[i][j])
            count += 1


def o_hidden1():
    variable.Output_hidden_1.clear()
    for i in range(variable.NUM_NODE_HIDDEN1):
        sum_output = 0
        for j in range(variable.NUM_NODE_INPUT):
            sum_output += variable.Output_input[j] * variable.Weight_input_hidden1[j][i]

        sum_output -= variable.THRESHOLD
        activation = sigmod(sum_output)
        variable.Output_hidden_1.insert(i, activation)


def o_hidden2():
    variable.Output_hidden_2.clear()
    for i in range(variable.NUM_NODE_HIDDEN2):
        sum_output = 0
        for j in range(variable.NUM_NODE_HIDDEN1):
            sum_output += variable.Output_hidden_1[j] * variable.Weight_hidden1_hidden2[j][i]

        sum_output -= variable.THRESHOLD
        activation = sigmod(sum_output)
        variable.Output_hidden_2.insert(i, activation)


def o_output():
    variable.Output_output.clear()
    for i in range(variable.NUM_NODE_OUTPUT):
        sum_output = 0
        for j in range(variable.NUM_NODE_HIDDEN2):
            sum_output += variable.Output_hidden_2[j] * variable.Weight_hidden2_output[j][i]

        sum_output -= variable.THRESHOLD
        activation = sigmod(sum_output)
        variable.Output_output.insert(i, activation)


def hyperbolic_tangent(number):
    try:
        result = ((2 * variable.A) / (1 + numpy.exp(-variable.B * number))) - variable.A
        #result = 2*sigmod(-2*number) - 1
    except OverflowError:
        result = 0
    return result


def error(id):
    variable.Error.clear()
    for i in range(variable.NUM_NODE_OUTPUT):
        variable.Error.insert(i, variable.expect_output[id][i] - variable.Output_output[i])


def d_output():
    variable.Delta_output.clear()
    for i in range(variable.NUM_NODE_OUTPUT):
        variable.Delta_output.insert(i, variable.Output_output[i] * (1 - variable.Output_output[i]) * variable.Error[i])


def d_hidden2():
    variable.Delta_hidden2.clear()
    for i in range(variable.NUM_NODE_HIDDEN2):
        sum_delta = 0
        for j in range(variable.NUM_NODE_OUTPUT):
            sum_delta += variable.Weight_hidden2_output[i][j] * variable.Delta_output[j]
        variable.Delta_hidden2.insert(i, variable.Output_hidden_2[i] * (1 - variable.Output_hidden_2[i]) * sum_delta)


def d_hidden1():
    variable.Delta_hidden1.clear()
    for i in range(variable.NUM_NODE_HIDDEN1):
        sum_delta = 0
        for j in range(variable.NUM_NODE_HIDDEN2):
            sum_delta += variable.Weight_hidden1_hidden2[i][j] * variable.Delta_hidden2[j]
        variable.Delta_hidden1.insert(i, variable.Output_hidden_1[i] * (1 - variable.Output_hidden_1[i]) * sum_delta)


def update_delta_weight_hidden2_output():
    for i in range(variable.NUM_NODE_HIDDEN2):
        for j in range(variable.NUM_NODE_OUTPUT):
            delta_weight = variable.ALPHA * variable.Output_hidden_2[i] * variable.Delta_output[j] + variable.BETA * variable.DW_hidden2_output[i][j]
            variable.DW_hidden2_output[i][j] = delta_weight


def update_delta_weight_hidden1_hidden2():
    for i in range(variable.NUM_NODE_HIDDEN1):
        for j in range(variable.NUM_NODE_HIDDEN2):
            delta_weight = variable.ALPHA * variable.Output_hidden_1[i] * variable.Delta_hidden2[j] + variable.BETA * variable.DW_hidden1_hidden2[i][j]
            variable.DW_hidden1_hidden2[i][j] = delta_weight


def update_delta_weight_input_hidden1():
    for i in range(variable.NUM_NODE_INPUT):
        for j in range(variable.NUM_NODE_HIDDEN1):
            delta_weight = variable.ALPHA * variable.Output_input[i] * variable.Delta_hidden1[j] + variable.BETA * variable.DW_input_hidden1[i][j]
            variable.DW_input_hidden1[i][j] = delta_weight


def update_weight_hidden2_output():
    for i in range(variable.NUM_NODE_HIDDEN2):
        for j in range(variable.NUM_NODE_OUTPUT):
            new_weight = variable.Weight_hidden2_output[i][j] + variable.DW_hidden2_output[i][j]
            #new_weight = variable.Weight_hidden2_output[i][j] + variable.ALPHA * variable.Output_hidden_2[i] * variable.Delta_output[j]
            variable.Weight_hidden2_output[i][j] = new_weight


def update_weight_hidden1_hidden2():
    for i in range(variable.NUM_NODE_HIDDEN1):
        for j in range(variable.NUM_NODE_HIDDEN2):
            new_weight = variable.Weight_hidden1_hidden2[i][j] + variable.DW_hidden1_hidden2[i][j]
            #new_weight = variable.Weight_hidden1_hidden2[i][j] + variable.ALPHA * variable.Output_hidden_1[i] * variable.Delta_hidden2[j]
            variable.Weight_hidden1_hidden2[i][j] = new_weight


def update_weight_input_hidden1():
    for i in range(variable.NUM_NODE_INPUT):
        for j in range(variable.NUM_NODE_HIDDEN1):
            new_weight = variable.Weight_input_hidden1[i][j] + variable.DW_input_hidden1[i][j]
            #new_weight = variable.Weight_input_hidden1[i][j] + variable.ALPHA * variable.Output_input[i] * variable.Delta_hidden1[j]
            variable.Weight_input_hidden1[i][j] = new_weight


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
    update_delta_weight_hidden1_hidden2()
    update_delta_weight_input_hidden1()
    update_weight_hidden2_output()
    update_weight_hidden1_hidden2()
    update_weight_input_hidden1()


#Utility Function
def counting(list):
    count = 0
    for i in range(len(list)):
        for j in range(len(list[i])):
            count+=1

    print(count)


def print_list():
    print("Output input")
    print(variable.Output_input)
    print("Output hidden1")
    print(variable.Output_hidden_1)
    print("Output hidden2")
    print(variable.Output_hidden_2)
    print("Output output")
    print(variable.Output_output)
    print("Error")
    print(variable.Error)
    print("Delta output")
    print(variable.Delta_output)
    print("Delta hidden 2")
    print(variable.Delta_hidden2)
    print("Delta hidden 1")
    print(variable.Delta_hidden1)
    print("Delta weight input to hidden1")
    print(variable.DW_input_hidden1)
    print("Delta weight hidden1 to hidden2")
    print(variable.DW_hidden1_hidden2)
    print("Delta weight hidden2 to output")
    print(variable.DW_hidden2_output)
    print("Weight input to hidden1")
    print(variable.Weight_input_hidden1)
    print("Weight hidden1 to hidden2")
    print(variable.Weight_hidden1_hidden2)
    print("Weight hidden2 to Output")
    print(variable.Weight_hidden2_output)


def leaky_relu(number):
    if number >= 0:
        return number
    else:
        return 0


def sigmod(number):
    return 1 / (1 + numpy.exp(-number))


def main():

    create_training_data_set(variable.image_A, variable.Train_data_set_A)
    init_weight()

    for i in range(100):
        print(i)
        for j in range(20):
            print("=================================" + str(i) + "----" + str(j) + "==================================")
            training_nn(0, variable.Train_data_set_A, j)
            print_list()

    image = scramble(variable.image_A, variable.RATE)
    o_input(image)
    o_hidden1()
    o_hidden2()
    o_output()
    print(variable.Output_output)


#main()