import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def total_variation_loss(image):
    batch_size, _, height, width = image.size()
    tv_h = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (batch_size * height * width)

def leakage_from_gradients(model_t, grads,  dummy_data_init, dummy_label_init):
    dummy_data = dummy_data_init
    dummy_label = dummy_label_init
    # optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    optimizer = torch.optim.Adam([dummy_data, dummy_label],lr=0.1)
    grad_diff_list = []
    if dummy_data.ndim == 3:
        dummy_data = dummy_data[:, None, :, :]
    else:
        dummy_data = dummy_data.permute(0, 3, 1, 2)
        # dummy_label = dummy_label.type(torch.LongTensor)
    grads = {k: v for k, v in grads.items() if
             'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k}

    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred_t = model_t(dummy_data)
            dummy_loss_t = F.cross_entropy(dummy_pred_t, dummy_label)
            dummy_grad_t = grad(dummy_loss_t, model_t.parameters(), create_graph=True)
            grad_diff = 0

            # for gx, gy in zip(dummy_grad_t, grads):
            #     grad_diff += ((gx - grads[gy]) ** 2).sum()

            tmp = 0
            norm1 = 0
            norm2 = 0

            layer_flag = 0
            if layer_flag == 0:
                for gx, gy in zip(dummy_grad_t, grads):
                    tmp += (gx * grads[gy]).sum()
                    norm1 += (gx * gx).sum()
                    norm2 += (grads[gy] * grads[gy]).sum()
            else:
                gx=dummy_grad_t[6]
                gy=grads['fc1.weight']
                tmp += (gx * gy).sum()
                norm1 += (gx * gx).sum()
                norm2 += (gy * gy).sum()

            grad_diff = 1 - tmp / torch.sqrt(norm1) / torch.sqrt(norm2)
            grad_diff += 0.01 * total_variation_loss(dummy_data)

            grad_diff.backward()
            grad_diff_list.append(grad_diff.detach().cpu().numpy())
            return grad_diff
        optimizer.step(closure)
    return dummy_data, dummy_label, grad_diff_list
