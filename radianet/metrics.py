import torch

def binary_accuracy(output, target, theshold =0.5 ):
    output[output > theshold] = 1
    output[output <= theshold] = 0
    print('out', output)
    print('target', target)

    acc = len(output[output == target])/len(output)
    return torch.tensor(acc*100)