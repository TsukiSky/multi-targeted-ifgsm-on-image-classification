"""
Targeted Attack using Iterative Fast Gradient Sign Method
"""

from attack.itfgsm import Itfgsm
import torch
import torch.nn as nn
import torch.nn.functional as F
# itfgsm = Itfgsm(model, epsilon=0.01, iters=40, target=True)


def itfgsm_attack(image, epsilon, model, orig_class, target_class, device, iter_num=10):
    # Skip if epsilon is 0
    if epsilon == 0:
        return image, False, 0
    worked = False

    for i in range(iter_num):
        # Zero out previous gradients
        image.grad = None
        # Forward pass
        out = model(image)
        # Calculate loss
        pred_loss = F.nll_loss(out, target_class)

        # Do backward pass and retain graph
        # pred_loss.backward()
        pred_loss.backward(retain_graph=True)

        # Add noise to processed image
        eps_image = image - epsilon*torch.sign(image.grad.data)
        eps_image.retain_grad()

        # Clipping eps_image to maintain pixel values into the [0, 1] range
        eps_image = torch.clamp(eps_image, 0, 1)

        # Forward pass
        new_output = model(eps_image)
        # Get prediction
        _, new_label = new_output.data.max(1)

        # Check if the new_label matches target, if so stop
        if new_label == target_class:
            worked = True
            break
        else:
            image = eps_image
            image.retain_grad()

    return eps_image, worked, i
