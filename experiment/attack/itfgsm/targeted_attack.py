"""
Targeted Attack using Iterative Fast Gradient Sign Method
"""

from attack.itfgsm import Itfgsm

itfgsm = Itfgsm(model, epsilon=0.01, iters=40, target=True)

