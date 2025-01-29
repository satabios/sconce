import torch

def pytorch_softmax(input_data, dim= -1):
    # As Large values of input may lead to overflow;
    # We subtract the largest value out of the input data
    # Before applying Softmax.
    # Note: Subtracting the same value from all the input data
    #       does not change the softmax result!
    input_max = torch.max(input_data, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(input_data- input_max)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x/sum_exp_x
