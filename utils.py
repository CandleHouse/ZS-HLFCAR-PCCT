import numpy as np
import cv2
import torch
import torch.nn.functional as F


def derivationHorizontal(img):
    kernel_x = np.array([[0, 0, 0],
                         [-1, 0, 1],
                         [0, 0, 0]])
    return np.squeeze(cv2.filter2D(img, -1, kernel_x) / 2)


def derivationVertical(img):
    kernel_y = np.array([[0, -1, 0],
                         [0, 0, 0],
                         [0, 1, 0]])
    return np.squeeze(cv2.filter2D(img, -1, kernel_y) / 2)


def duTensor(img: torch.FloatTensor, device):
    B, C, H, W = img.shape
    kernel_x = torch.FloatTensor([[0, 0, 0],
                                 [-1, 0, 1],
                                 [0, 0, 0]]).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1) / 2
    weight = torch.nn.Parameter(data=kernel_x, requires_grad=False).to(device)
    x = F.conv2d(img.clone(), weight, padding=1)
    return x


def dvTensor(img: torch.FloatTensor, device):
    B, C, H, W = img.shape
    kernel_y = torch.FloatTensor([[0, -1, 0],
                                  [0, 0, 0],
                                  [0, 1, 0]]).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1) / 2
    weight = torch.nn.Parameter(data=kernel_y, requires_grad=False).to(device)
    y = F.conv2d(img.clone(), weight, padding=1)
    return y


def ScharrXTensor3(img: torch.FloatTensor, device):
    B, C, H, W = img.shape
    kernel_x = torch.FloatTensor([[-3, 0, 3],
                                  [-10, 0, 10],
                                  [-3, 0, 3]]).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
    weight = torch.nn.Parameter(data=kernel_x, requires_grad=False).to(device)
    x = F.conv2d(img.clone(), weight, padding=1)
    return x


def SobelXTensor5(img: torch.FloatTensor, device):
    B, C, H, W = img.shape
    kernel_x = torch.FloatTensor([[-1, -2, 0, 2, 1],
                                  [-4, -8, 0, 8, 4],
                                  [-6, -12, 0, 12, 6],
                                  [-4, -8, 0, 8, 4],
                                  [-1, -2, 0, 2, 1]]).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
    weight = torch.nn.Parameter(data=kernel_x, requires_grad=False).to(device)
    x = F.conv2d(img.clone(), weight, padding=1)
    return x


def take(x):
    return x.squeeze().detach().cpu().numpy()
