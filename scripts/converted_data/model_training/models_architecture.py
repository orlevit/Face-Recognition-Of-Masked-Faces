import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from config import EMBBEDINGS_NUMBER, MODELS_NUMBER, EMBBEDINGS_REDUCED

class NeuralNetwork1(nn.Module):
    def __init__(self):
        super(NeuralNetwork1, self).__init__()
        self.flatten = nn.Flatten()
        self.bilinar = nn.Bilinear(EMBBEDINGS_NUMBER * MODELS_NUMBER, EMBBEDINGS_NUMBER * MODELS_NUMBER, 1)

    def forward(self, emb1, emb2):
        emb1 = self.flatten(emb1)
        emb2 = self.flatten(emb2)
        logits = self.bilinar(emb1, emb2)
        return logits


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.bilinar = nn.Bilinear(EMBBEDINGS_REDUCED * MODELS_NUMBER, EMBBEDINGS_REDUCED * MODELS_NUMBER, 1)

    def forward(self, emb1, emb2):
        reduce_dim1 = torch.cat([self.reduction(emb1[:, model_i, :]) for model_i in range(MODELS_NUMBER)], axis = -1)
        reduce_dim2 = torch.cat([self.reduction(emb2[:, model_i, :]) for model_i in range(MODELS_NUMBER)], axis = -1)
        logits = self.bilinar(reduce_dim1, reduce_dim2)
        return logits


class NeuralNetwork3(nn.Module):
    def __init__(self):
        super(NeuralNetwork3, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.bilinar = nn.Bilinear(EMBBEDINGS_REDUCED * MODELS_NUMBER, EMBBEDINGS_REDUCED * MODELS_NUMBER, 1)

    def forward(self, emb1, emb2): 
        red1_1 = F.relu(self.reduction1(emb1[:, 0, :]))
        red2_1 = F.relu(self.reduction2(emb1[:, 1, :]))
        red3_1 = F.relu(self.reduction3(emb1[:, 2, :]))
        red4_1 = F.relu(self.reduction4(emb1[:, 3, :]))
        red5_1 = F.relu(self.reduction5(emb1[:, 4, :]))
        red6_1 = F.relu(self.reduction6(emb1[:, 5, :]))
        red7_1 = F.relu(self.reduction7(emb1[:, 6, :]))

        red1_2 = F.relu(self.reduction1(emb2[:, 0, :]))
        red2_2 = F.relu(self.reduction2(emb2[:, 1, :]))
        red3_2 = F.relu(self.reduction3(emb2[:, 2, :]))
        red4_2 = F.relu(self.reduction4(emb2[:, 3, :]))
        red5_2 = F.relu(self.reduction5(emb2[:, 4, :]))
        red6_2 = F.relu(self.reduction6(emb2[:, 5, :]))
        red7_2 = F.relu(self.reduction7(emb2[:, 6, :]))

        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        logits = self.bilinar(reduce_dim1, reduce_dim2)
        return logits

class NeuralNetwork3_2(nn.Module):
    def __init__(self):
        super(NeuralNetwork3_2, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.last = nn.Linear(2*EMBBEDINGS_REDUCED * MODELS_NUMBER, EMBBEDINGS_REDUCED * MODELS_NUMBER, 1)

    def forward(self, emb1, emb2): 
        red1_1 = F.relu(self.reduction1(emb1[:, 0, :]))
        red2_1 = F.relu(self.reduction2(emb1[:, 1, :]))
        red3_1 = F.relu(self.reduction3(emb1[:, 2, :]))
        red4_1 = F.relu(self.reduction4(emb1[:, 3, :]))
        red5_1 = F.relu(self.reduction5(emb1[:, 4, :]))
        red6_1 = F.relu(self.reduction6(emb1[:, 5, :]))
        red7_1 = F.relu(self.reduction7(emb1[:, 6, :]))

        red1_2 = F.relu(self.reduction1(emb2[:, 0, :]))
        red2_2 = F.relu(self.reduction2(emb2[:, 1, :]))
        red3_2 = F.relu(self.reduction3(emb2[:, 2, :]))
        red4_2 = F.relu(self.reduction4(emb2[:, 3, :]))
        red5_2 = F.relu(self.reduction5(emb2[:, 4, :]))
        red6_2 = F.relu(self.reduction6(emb2[:, 5, :]))
        red7_2 = F.relu(self.reduction7(emb2[:, 6, :]))

        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        concat = torch.cat([reduce_dim1, reduce_dim2], axis = -1)
        logits = self.last(concat)
        return logits

class NeuralNetwork4(nn.Module):
    def __init__(self):
        super(NeuralNetwork4, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.fc1 = nn.Linear(2 * EMBBEDINGS_REDUCED * MODELS_NUMBER, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])

        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])

        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        logits = self.fc2(x)
        return logits


class NeuralNetwork5(nn.Module):
    def __init__(self):
        super(NeuralNetwork5, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.fc1 = nn.Linear(3 * EMBBEDINGS_REDUCED * MODELS_NUMBER, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])

        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])

        reduce_corr = torch.cat([red1_1 * red1_2, red2_1 * red2_2, red3_1 * red3_2, red4_1 * red4_2, red5_1 * red5_2, red6_1 * red6_2, red7_1 * red7_2], axis = -1)
        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2, reduce_corr], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        logits = self.fc2(x)
        return logits

class NeuralNetwork5_2(nn.Module):
    def __init__(self):
        super(NeuralNetwork5_2, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.fc1 = nn.Linear(EMBBEDINGS_REDUCED * MODELS_NUMBER, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])

        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])

        reduce_corr = torch.cat([red1_1 * red1_2, red2_1 * red2_2, red3_1 * red3_2, red4_1 * red4_2, red5_1 * red5_2, red6_1 * red6_2, red7_1 * red7_2], axis = -1)
        x = torch.tanh(self.fc1(reduce_corr))
        logits = self.fc2(x)
        return logits

class NeuralNetwork5_3(nn.Module):
    def __init__(self):
        super(NeuralNetwork5_3, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.fc1 = nn.Linear(3 * EMBBEDINGS_REDUCED * MODELS_NUMBER, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])

        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])

        reduce_corr = torch.cat([red1_1 * red1_2, red2_1 * red2_2, red3_1 * red3_2, red4_1 * red4_2, red5_1 * red5_2, red6_1 * red6_2, red7_1 * red7_2], axis = -1)
        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2, reduce_corr], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

class NeuralNetwork5_4(nn.Module):
    def __init__(self):
        super(NeuralNetwork5_4, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.fc1 = nn.Linear(EMBBEDINGS_REDUCED * MODELS_NUMBER, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])

        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])

        reduce_corr = torch.cat([red1_1 * red1_2, red2_1 * red2_2, red3_1 * red3_2, red4_1 * red4_2, red5_1 * red5_2, red6_1 * red6_2, red7_1 * red7_2], axis = -1)
        x = torch.tanh(self.fc1(reduce_corr))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

class NeuralNetwork5_5(nn.Module):
    def __init__(self):
        super(NeuralNetwork5_5, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.fc1 = nn.Linear(2 * EMBBEDINGS_REDUCED * MODELS_NUMBER, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])

        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])

        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

class NeuralNetwork5_6(nn.Module):
    def __init__(self):
        super(NeuralNetwork5_6, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.fc1 = nn.Linear(49 * EMBBEDINGS_REDUCED , 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])

        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])

        reduce_dim1_list = [red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1]
        reduce_dim2_list = [red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2]
        all_corr_list = [red1*red2 for red1 in reduce_dim1_list for red2 in reduce_dim2_list]

        all_corr = torch.cat(all_corr_list, axis = -1)
        x = torch.tanh(self.fc1(all_corr))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

class NeuralNetwork6(nn.Module):
    def __init__(self):
        super(NeuralNetwork6, self).__init__()
        self.reduction = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.bilinar = nn.Bilinear(EMBBEDINGS_REDUCED , EMBBEDINGS_REDUCED , 2)

    def forward(self, emb1, emb2):
        norm1 = torch.from_numpy(normalize(emb1[:, 0, :])).float()
        norm2 = torch.from_numpy(normalize(emb2[:, 0, :])).float()
        #import pdb; pdb.set_trace();
        reduction1 = self.reduction(norm1)
        reduction2 = self.reduction(norm2)
        logits = self.bilinar(reduction1, reduction2)
        return logits

class NeuralNetwork7(nn.Module):
    def __init__(self):
        super(NeuralNetwork7, self).__init__()
        self.thr = nn.Linear(1, 1)

    def forward(self, emb1, emb2):
        norm1 = normalize(emb1[:, 0, :])
        norm2 = normalize(emb2[:, 0, :])
        diff = np.subtract(norm1, norm2)
        dist = np.sum(np.square(diff), 1)
        dist_torch = torch.from_numpy(np.expand_dims(dist, 1))
        logits = self.thr(dist_torch.type(torch.float))
        return logits

class NeuralNetwork8(nn.Module):
    def __init__(self):
        super(NeuralNetwork8, self).__init__()
        self.hidden1 = nn.Linear(1, 16)
        self.thr = nn.Linear(16, 1)

    def forward(self, emb1, emb2):
        norm1 = normalize(emb1[:, 0, :])
        norm2 = normalize(emb2[:, 0, :])
        diff = np.subtract(norm1, norm2)
        dist = np.sum(np.square(diff), 1)
        dist_torch = torch.from_numpy(np.expand_dims(dist, 1))
        x = self.hidden1(dist_torch.type(torch.float))
        x = torch.tanh(x)
        logits = self.thr(x)
        return logits

class NeuralNetwork82(nn.Module):
    def __init__(self):
        super(NeuralNetwork82, self).__init__()
        self.hidden1 = nn.Linear(1, 64)
        self.thr = nn.Linear(64, 1)

    def forward(self, emb1, emb2):
        norm1 = normalize(emb1[:, 0, :])
        norm2 = normalize(emb2[:, 0, :])
        diff = np.subtract(norm1, norm2)
        dist = np.sum(np.square(diff), 1)
        dist_torch = torch.from_numpy(np.expand_dims(dist, 1))
        x = self.hidden1(dist_torch.type(torch.float))
        x = torch.tanh(x)
        logits = self.thr(x)
        return logits

class NeuralNetwork83(nn.Module):
    def __init__(self):
        super(NeuralNetwork83, self).__init__()
        self.hidden1 = nn.Linear(1, 64)
        self.hidden2 = nn.Linear(64, 16)
        self.thr = nn.Linear(16, 1)

    def forward(self, emb1, emb2):
        norm1 = normalize(emb1[:, 0, :])
        norm2 = normalize(emb2[:, 0, :])
        diff = np.subtract(norm1, norm2)
        dist = np.sum(np.square(diff), 1)
        dist_torch = torch.from_numpy(np.expand_dims(dist, 1))
        x = self.hidden1(dist_torch.type(torch.float))
        x = torch.tanh(x)
        x = self.hidden2(x)
        x = torch.tanh(x)
        logits = self.thr(x)
        return logits

class NeuralNetwork84(nn.Module):
    def __init__(self):
        super(NeuralNetwork84, self).__init__()
        self.hidden1 = nn.Linear(1, 70)
        self.hidden2 = nn.Linear(70, 60)
        self.hidden3 = nn.Linear(60, 50)
        self.hidden4 = nn.Linear(50, 40)
        self.hidden5 = nn.Linear(40, 30)
        self.hidden6 = nn.Linear(30, 20)
        self.thr = nn.Linear(20, 1)

    def forward(self, emb1, emb2):
        norm1 = normalize(emb1[:, 0, :])
        norm2 = normalize(emb2[:, 0, :])
        diff = np.subtract(norm1, norm2)
        dist = np.sum(np.square(diff), 1)
        dist_torch = torch.from_numpy(np.expand_dims(dist, 1))
        x = self.hidden1(dist_torch.type(torch.float))
        x = F.dropout(torch.tanh(x),p=0.8)
        x = self.hidden2(x)
        x = F.dropout(torch.tanh(x),p=0.8)
        x = self.hidden3(x)
        x = F.dropout(torch.tanh(x),p=0.8)
        x = self.hidden4(x)
        x = F.dropout(torch.tanh(x),p=0.8)
        x = self.hidden5(x)
        x = F.dropout(torch.tanh(x),p=0.8)
        x = self.hidden6(x)
        x = F.dropout(torch.tanh(x),p=0.8)
        logits = self.thr(x)
        return logits

class LogisticRegressionOr(nn.Module):
     def __init__(self):
        super(LogisticRegressionOr, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

     def forward(self, emb1, emb2):
        norm1 = normalize(emb1[:, 0, :])
        norm2 = normalize(emb2[:, 0, :])
        diff = np.subtract(norm1, norm2)
        dist = np.sum(np.square(diff), 1)
        dist_torch = torch.from_numpy(np.expand_dims(dist, 1))
        outputs = torch.sigmoid(self.linear(dist_torch.type(torch.float)))
        return outputs

class NeuralNetwork9(nn.Module):
    def __init__(self):
        super(NeuralNetwork9, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.cos_sim = nn.CosineSimilarity()
        self.fc1 = nn.Linear(2 * EMBBEDINGS_REDUCED * MODELS_NUMBER + MODELS_NUMBER, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])
        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])
        
        cs0 = self.cos_sim(emb1[:, 0, :], emb2[:, 0, :])[:, None]
        cs1 = self.cos_sim(emb1[:, 1, :], emb2[:, 1, :])[:, None]
        cs2 = self.cos_sim(emb1[:, 2, :], emb2[:, 2, :])[:, None]
        cs3 = self.cos_sim(emb1[:, 3, :], emb2[:, 3, :])[:, None]
        cs4 = self.cos_sim(emb1[:, 4, :], emb2[:, 4, :])[:, None]
        cs5 = self.cos_sim(emb1[:, 5, :], emb2[:, 5, :])[:, None]
        cs6 = self.cos_sim(emb1[:, 6, :], emb2[:, 6, :])[:, None]

        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2, cs0, cs1, cs2, cs3, cs4, cs5, cs6], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        logits = self.fc2(x)
        return logits

class NeuralNetwork10(nn.Module):
    def __init__(self):
        super(NeuralNetwork10, self).__init__()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.cos_sim = nn.CosineSimilarity()
        self.fc1 = nn.Linear(2 * EMBBEDINGS_REDUCED * MODELS_NUMBER, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])
        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])
        
        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        logits = self.fc2(x)
        return logits

class NeuralNetwork11(nn.Module):
    def __init__(self):
        super(NeuralNetwork11, self).__init__()
        self.fc1 = nn.Linear(2 * 512 * MODELS_NUMBER, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, emb1, emb2): 
        red1_1 = emb1[:, 0, :]
        red2_1 = emb1[:, 1, :]
        red3_1 = emb1[:, 2, :]
        red4_1 = emb1[:, 3, :]
        red5_1 = emb1[:, 4, :]
        red6_1 = emb1[:, 5, :]
        red7_1 = emb1[:, 6, :]
        red1_2 = emb2[:, 0, :]
        red2_2 = emb2[:, 1, :]
        red3_2 = emb2[:, 2, :]
        red4_2 = emb2[:, 3, :]
        red5_2 = emb2[:, 4, :]
        red6_2 = emb2[:, 5, :]
        red7_2 = emb2[:, 6, :]
        
        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        logits = self.fc2(x)
        return logits


class NeuralNetwork12(nn.Module):
    def __init__(self):
        super(NeuralNetwork12, self).__init__()
        cross_number = MODELS_NUMBER * (MODELS_NUMBER -1)
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.bilinears = nn.ModuleList([nn.Bilinear(32 , 32 , 64, bias=True)  for i in range(cross_number)])
        self.fc1 = nn.Linear(64 * cross_number, 64)
        self.fc2 = nn.Linear(64, 1)

    def _get_binilinar_list(self, reduce_dim1_list, reduce_dim2_list):
        idx = 0
        bilinar_list = []
        for i, red1 in enumerate(reduce_dim1_list, 1):
            for j, red2 in enumerate(reduce_dim2_list, 1):
                if i != j:
                   bilinar_list.append(self.bilinears[idx](red1, red2))
                   idx += 1
        return bilinar_list           

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])
        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])
        
        reduce_dim1_list = [red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1]
        reduce_dim2_list = [red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2]
        bilinar_list = self._get_binilinar_list(reduce_dim1_list, reduce_dim2_list)
        bilinars_concat = torch.cat(bilinar_list, axis = -1)
        x = torch.tanh(self.fc1(bilinars_concat))
        logits = self.fc2(x)
        return logits

class NeuralNetwork13(nn.Module):
    def __init__(self):
        super(NeuralNetwork13, self).__init__()
        cross_number = MODELS_NUMBER * (MODELS_NUMBER -1)
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, EMBBEDINGS_REDUCED)
        self.bilinears = nn.ModuleList([nn.Bilinear(32 , 32 , 64, bias=True)  for i in range(cross_number)])
        self.fc1 = nn.Linear(64 * cross_number + 32 * MODELS_NUMBER * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def _get_binilinar_list(self, reduce_dim1_list, reduce_dim2_list):
        idx = 0
        bilinar_list = []
        for i, red1 in enumerate(reduce_dim1_list, 1):
            for j, red2 in enumerate(reduce_dim2_list, 1):
                if i != j:
                   bilinar_list.append(self.bilinears[idx](red1, red2))
                   idx += 1
        return bilinar_list           

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])
        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])
        
        reduce_dim1_list = [red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1]
        reduce_dim2_list = [red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2]
        bilinar_list = self._get_binilinar_list(reduce_dim1_list, reduce_dim2_list)
        bilinars_concat = torch.cat(bilinar_list, axis = -1)
        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2, bilinars_concat], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        logits = self.fc2(x)
        return logits

class NeuralNetwork14(nn.Module):
    def __init__(self):
        super(NeuralNetwork14, self).__init__()
        self.flatten = nn.Flatten()
        self.reduction1 = nn.Linear(EMBBEDINGS_NUMBER, 4)
        self.reduction2 = nn.Linear(EMBBEDINGS_NUMBER, 4)
        self.reduction3 = nn.Linear(EMBBEDINGS_NUMBER, 4)
        self.reduction4 = nn.Linear(EMBBEDINGS_NUMBER, 4)
        self.reduction5 = nn.Linear(EMBBEDINGS_NUMBER, 4)
        self.reduction6 = nn.Linear(EMBBEDINGS_NUMBER, 4)
        self.reduction7 = nn.Linear(EMBBEDINGS_NUMBER, 4)
        self.cos_sim = nn.CosineSimilarity()
        self.fc1 = nn.Linear(2 * 4 * MODELS_NUMBER + MODELS_NUMBER, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, emb1, emb2): 
        red1_1 = self.reduction1(emb1[:, 0, :])
        red2_1 = self.reduction2(emb1[:, 1, :])
        red3_1 = self.reduction3(emb1[:, 2, :])
        red4_1 = self.reduction4(emb1[:, 3, :])
        red5_1 = self.reduction5(emb1[:, 4, :])
        red6_1 = self.reduction6(emb1[:, 5, :])
        red7_1 = self.reduction7(emb1[:, 6, :])
        red1_2 = self.reduction1(emb2[:, 0, :])
        red2_2 = self.reduction2(emb2[:, 1, :])
        red3_2 = self.reduction3(emb2[:, 2, :])
        red4_2 = self.reduction4(emb2[:, 3, :])
        red5_2 = self.reduction5(emb2[:, 4, :])
        red6_2 = self.reduction6(emb2[:, 5, :])
        red7_2 = self.reduction7(emb2[:, 6, :])
        
        cs0 = self.cos_sim(emb1[:, 0, :], emb2[:, 0, :])[:, None]
        cs1 = self.cos_sim(emb1[:, 1, :], emb2[:, 1, :])[:, None]
        cs2 = self.cos_sim(emb1[:, 2, :], emb2[:, 2, :])[:, None]
        cs3 = self.cos_sim(emb1[:, 3, :], emb2[:, 3, :])[:, None]
        cs4 = self.cos_sim(emb1[:, 4, :], emb2[:, 4, :])[:, None]
        cs5 = self.cos_sim(emb1[:, 5, :], emb2[:, 5, :])[:, None]
        cs6 = self.cos_sim(emb1[:, 6, :], emb2[:, 6, :])[:, None]

        reduce_dim1 = torch.cat([red1_1, red2_1, red3_1, red4_1, red5_1, red6_1, red7_1], axis = -1)
        reduce_dim2 = torch.cat([red1_2, red2_2, red3_2, red4_2, red5_2, red6_2, red7_2], axis = -1)
        reduce_dim = torch.cat([reduce_dim1, reduce_dim2, cs0, cs1, cs2, cs3, cs4, cs5, cs6], axis = -1)
        x = torch.tanh(self.fc1(reduce_dim))
        logits = self.fc2(x)
        return logits

