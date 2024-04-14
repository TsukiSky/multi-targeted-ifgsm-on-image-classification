import matplotlib.pyplot as plt
import numpy as np


models = ['CNN', 'ResNet', 'ViT']
metrics = ['Hash Distance', 'L2 Distance', 'SSIM']
data_itfgsm = np.array([
    [0.314, 2.839830002069473, 0.9745711825489998],  # CNN
    [0.176, 2.10541999745369, 0.9861733474135399],  # ResNet
    [0.55, 3.829099921703339, 0.9557166385054588]  # ViT
])
data_mt_ifgsm = np.array([
    [0.3, 2.6385300002098084, 0.9772909125089645],  # CNN
    [0.188, 1.362319997906685, 0.9933819687962532],  # ResNet
    [0.56, 3.827579924106598, 0.9552195736765862]  # ViT
])

n_models = len(models)
index = np.arange(n_models)
bar_width = 0.35
opacity = 0.8

y_limits = [(None, None), (1, 4), (0.94, 1.0)]

for i, metric in enumerate(metrics):
    plt.figure(i)
    plt.bar(index, data_itfgsm[:, i], bar_width, alpha=opacity, color='#F27970', label='ITFGSM', hatch='/',
            edgecolor='black')
    plt.bar(index + bar_width, data_mt_ifgsm[:, i], bar_width, alpha=opacity, color='#54B345', label='MT-IFGSM',
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
