import matplotlib.pyplot as plt
import numpy as np

# Performance data
model_names = ['CNN', 'ResNet', 'ViT']
accuracy_data = np.array([0.000, 0.000, 0.000, 0.000, 0.581, 0.581])
hamming_loss_data = np.array([0.21293334052711727, 0.19926666965335607,
                              0.2148666743040085, 0.18960000357031823,
                              0.06906666938960553, 0.06906666938960553])

# Set the positions and width for the bars
positions = np.arange(len(model_names))
width = 0.35

# Define the color and hatch pattern
colors = ['#FFBE7A', '#8ECFC9']
patterns = ['/', '\\']

# Function to add value labels above bars
def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

# Plotting Accuracy
plt.figure(figsize=(8, 6))
bar1 = plt.bar(positions - width/2, accuracy_data[::2], width, label='ITFGSM', color=colors[0], hatch=patterns[0], edgecolor='black')
bar2 = plt.bar(positions + width/2, accuracy_data[1::2], width, label='MT-IFGSM', color=colors[1], hatch=patterns[1], edgecolor='black')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Models by Attack Type')
plt.xticks(positions, model_names)
add_value_labels(plt.gca(), bar1 + bar2)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting Hamming Loss
plt.figure(figsize=(8, 6))
bar3 = plt.bar(positions - width/2, hamming_loss_data[::2], width, label='ITFGSM', color=colors[0], hatch=patterns[0], edgecolor='black')
bar4 = plt.bar(positions + width/2, hamming_loss_data[1::2], width, label='MT-IFGSM', color=colors[1], hatch=patterns[1], edgecolor='black')
plt.xlabel('Models')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss of Models by Attack Type')
plt.xticks(positions, model_names)
add_value_labels(plt.gca(), bar3 + bar4)
plt.legend()
plt.tight_layout()
plt.show()
