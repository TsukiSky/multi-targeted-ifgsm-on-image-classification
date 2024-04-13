import matplotlib.pyplot as plt
import numpy as np


models = ['CNN', 'ResNet', 'ViT']
metrics = ['Hash Distance', 'L2 Distance', 'SSIM']
data_itfgsm = np.array([
    [0.334, 2.822330001831055, 1.0],  # CNN
    [0.182, 2.0541699966192244, 1.0],  # ResNet
    [0.55, 3.829099921703339, 1.0]  # ViT
])
data_mt_itfgsm = np.array([
    [0.286, 2.663689998626709, 0.9788],  # CNN
    [0.224, 1.3393899999856949, 0.9841],  # ResNet
    [0.56, 3.827579924106598, 0.9575]  # ViT
])

n_models = len(models)
index = np.arange(n_models)
bar_width = 0.35
opacity = 0.8

y_limits = [(None, None), (1, 4), (0.94, 1.02)]

for i, metric in enumerate(metrics):
    plt.figure(i)
    plt.bar(index, data_itfgsm[:, i], bar_width, alpha=opacity, color='#F27970', label='ITFGSM', hatch='/',
            edgecolor='black')
    plt.bar(index + bar_width, data_mt_itfgsm[:, i], bar_width, alpha=opacity, color='#54B345', label='MT-ITFGSM',
            hatch='\\', edgecolor='black')

    plt.xlabel('Models')
    plt.ylabel('Metric Values')
    plt.title(f'{metric} by Model and Attack Type')
    plt.xticks(index + bar_width / 2, models)
    plt.legend()
    if y_limits[i][0] is not None:
        plt.ylim(y_limits[i][0], y_limits[i][1])
    plt.tight_layout()

plt.show()
