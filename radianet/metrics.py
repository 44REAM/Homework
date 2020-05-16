import torch

def binary_accuracy(output, target, theshold =0.5, pos_weight = 1):
    output[output > theshold] = 1
    output[output <= theshold] = 0

    len_pos = len(target[target == 1])*pos_weight
    len_neg = len(target[target == 0])
    len_all = len_pos+len_neg

    true_pos = len(output[(output == target) & (target == 1)])*pos_weight
    true_neg = len(output[(output == target) & (target == 0)])

    acc = (true_pos+true_neg)/len_all
    return torch.tensor(acc*100)

def binary_confusion_matrix(output, target, theshold = 0.48):
    output[output > theshold] = 1
    output[output <= theshold] = 0

    print(confusion_matrix(output.cpu(), target.cpu()))

if __name__ == "__main__":
    output = torch.tensor([1,1,1,1,1,1])
    target = torch.tensor([1,1,1,0,0,0])
    result = binary_accuracy(output, target, pos_weight = 0.3)
    print(result)