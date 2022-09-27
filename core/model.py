import numpy as np
from random import random, randrange

import torch
import torch.nn as nn

import cv2

def to_binary(x):
    return 1 if round(x) == 1 else 0

class CNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNDQN, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)

    def act(self, state, device, epsilon=0.0):
        # cv2.imshow('state', np.uint8(state))
        # cv2.waitKey(1)
        state = torch.FloatTensor(np.expand_dims(np.float32(state).transpose(2, 1, 0), axis=0)).to(device)
        
        # print(state)
        q_value = self.forward(state).cpu().detach().numpy()
        # print(q_value)
        return {
                    "ESC": to_binary(q_value[0][0]),
                    "attack": to_binary(q_value[0][1]),
                    "back": to_binary(q_value[0][2]),
                    "camera": [q_value[0][3], q_value[0][4]],
                    "drop": to_binary(q_value[0][5]),
                    "forward": to_binary(q_value[0][6]),
                    "hotbar.1": to_binary(q_value[0][7]),
                    "hotbar.2": to_binary(q_value[0][8]),
                    "hotbar.3": to_binary(q_value[0][9]),
                    "hotbar.4": to_binary(q_value[0][10]),
                    "hotbar.5": to_binary(q_value[0][11]),
                    "hotbar.6": to_binary(q_value[0][12]),
                    "hotbar.7": to_binary(q_value[0][13]),
                    "hotbar.8": to_binary(q_value[0][14]),
                    "hotbar.9": to_binary(q_value[0][15]),
                    "inventory": to_binary(q_value[0][16]),
                    "jump": to_binary(q_value[0][17]),
                    "left": to_binary(q_value[0][18]),
                    "pickItem": to_binary(q_value[0][19]),
                    "right": to_binary(q_value[0][20]),
                    "sneak": to_binary(q_value[0][21]),
                    "sprint": to_binary(q_value[0][22]),
                    "swapHands": to_binary(q_value[0][23]),
                    "use": to_binary(q_value[0][24])
                }
        
