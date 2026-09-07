import numpy as np
import random
import math
import torch.nn.functional as F
import torch

import collections
import copy

class node(object):
    def __init__(self,neighbors, dataset, model,device,lr=1e-2):
        self.neighbor=np.asmatrix(neighbors)
        self.neighbor=self.neighbor.A
        self.dataloader=dataset

        self.model=model
        self.model.train()
        self.model.to(device)

        self.lr=lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        # self.optimizer=torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.6, 0.999))

    def active_update(self,device,epo=1,quantize=1e10,capture_gradients=False):
        captured_gradients = None
        for e in range(epo):
            img,label=next(iter(self.dataloader))
            if img.ndim==3:
                img=img[:,None,:,:]
            else:
                img=img.permute(0,3,1,2)
                label = label.type(torch.LongTensor)
            img,label=img.to(device),label.to(device)
            self.optimizer.zero_grad()
            output=self.model(img)
            loss=F.cross_entropy(output,label)

            loss.backward()
            if capture_gradients:
                captured_gradients = collections.OrderedDict(
                    (name, parameter.grad.detach().cpu().clone())
                    for name, parameter in self.model.named_parameters()
                )
            self.optimizer.step()

        quantized_state_dict = {}
        for param_name, param_tensor in self.model.state_dict().items():
            quantized_state_dict[param_name] = torch.round(param_tensor * quantize) / quantize
        self.model.load_state_dict(quantized_state_dict)

        # self.model.eval()
        # self.model.zero_grad()
        # ground_truth, label = next(iter(self.dataloader))
        # label = label.type(torch.LongTensor)
        # ground_truth=ground_truth.flatten().unsqueeze(0)
        # output = self.model(ground_truth)
        # loss = F.cross_entropy(output, label)
        # input_gradient = torch.autograd.grad(loss, self.model.parameters())
        # input_gradient = [grad.detach() for grad in input_gradient]
        # with torch.no_grad():
        #     for j, param in enumerate(self.model.parameters()):
        #         param.copy_(param - self.lr * input_gradient[j])

        # assert math.isnan(loss.item())==False,'NaN'
        if capture_gradients:
            return loss.item(), captured_gradients
        return loss.item()
