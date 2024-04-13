# this script is used to draw the graph of the performance of victim models
# refer to readme.md for more details
import matplotlib.pyplot as plt

vit_performance = {"accuracy": 0.5896, "hamming_loss": 0.0680}
resnet_performance = {"accuracy": 0.4176, "hamming_loss": 0.0721}
cnn_performance = {"accuracy": 0.3802, "hamming_loss": 0.0771}

models = ["2-layer CNN", "ResNet18", "ViT"]
accuracy = [cnn_performance["accuracy"], resnet_performance["accuracy"], vit_performance["accuracy"]]
hamming_loss = [cnn_performance["hamming_loss"], resnet_performance["hamming_loss"], vit_performance["hamming_loss"]]
patterns = ['//', '\\\\', '+']
colors = ['#2878b5', '#54B345', '#FA7F6F']


plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracy, color=colors, hatch=patterns, edgecolor='black')
plt.title('Models Accuracy')
plt.ylim(0, 0.7)
plt.ylabel('Accuracy')

for bar, pattern in zip(bars, patterns):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
# plt.figtext(0.5, 0.01, '[Figure 5]. Accuracy of models', ha='center')
plt.show()

plt.figure(figsize=(6, 4))
bars = plt.bar(models, hamming_loss, color=colors, hatch=patterns, edgecolor='black')
plt.title('Models Hamming Loss')
plt.ylim(0, 0.1)
plt.ylabel('Hamming Loss')

for bar, pattern in zip(bars, patterns):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
# plt.figtext(0.5, 0.01, '[Figure 6]. Hamming loss of models', ha='center')
plt.show()
