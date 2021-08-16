import torchvision
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from DenseNet import *

mat = sio.loadmat('./datasets/train_32x32.mat')
img = [[[mat['X'][i][j][k][20000] for k in range(3)] for j in range(32)] for i in range(32)]
label = mat['y'][20000]
plt.figure()
plt.imshow(img)
plt.title('Label = %s' %label)
plt.axis('off')

nr_classes = 10
depth = 100
model = DenseNet(depth , nr_classes)

conv_layers = []
model_children = list(model.children())
for i in range(len(model_children)-3):
    conv_layers.append(model_children[i])

transform = torchvision.transforms.ToTensor()
img = transform(np.array(img))
img = img.unsqueeze(0)

results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[i-1]))
outputs = results

for num_layer in range(len(outputs)):
    plt.figure(figsize=(10,10))
    layer_viz = outputs[num_layer][0,:,:,:]
    layer_viz = layer_viz.data
    plt.suptitle('Feature maps size = %s' %list(layer_viz.size()))

    for i, filter in enumerate(layer_viz):
        if i == 9:
            break
        plt.subplot(3, 3, i+1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')

plt.show()