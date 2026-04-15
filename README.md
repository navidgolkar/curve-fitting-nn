# curve-fitting-nn
4 different configurations of neural networks used for curve fitting + visualization

in main.py you can change:
- H_N: number of hidden layers
- N_N: number of nodes in each hidden layer
- FUNC: actiovation function (the same function is used throughout the network)
- EPOCHS: number of epochs for training
- LR: learning rate
- LOG_EVERY: the intervals in which the figures show the animation
- K_SIZE, PADDING, STRIDE: kernel size, padding, and stride for CNN and ConvResNet
- CONNECT: connection number for ConvResNet (how many residual feedforward connections should there be in each hidden layer, but if the number of connections can't be reached taht layer won't have any residual connections)
- INPUT_N: number of inputs (the range of 0 to 4 would be devided to that many inputs and outputs)
- NOISE_STD: noise standard deviation (for better comparison of networks, here it's 0.1)
- SHOW: boolean to indicated whether the figures should be shown when the run is done
- FILE_TYPE: file type, i recommend 'gif' or 'png' (other that gif which uses FuncAnimation of matplotlib.animation, other filetypes will be used through savefig so any file type supported by matplotlib figures will work), and if it's set to empty ("") there will be no saved file
- SAVE: savepath for the animations or images of output

in the console the mean squared loss, and binary cross entropy loss at each epoch (based on LOG_EVERY) is written.
in the saved file 4 plots are drawn (visualization here shown is a file saved as '.gif'):
- Top left: data vs prediction
- Top Right: the graph of network (red connections mean negative weights, blue means positive, and opacity is based on the absolute value of the weights, so smaller weight, lower opacity)
- Bottom Left: Mean Squared Loss through epochs
- Bottom Right: Binary Cross Entropy Loss through epochs

to use you can clone this repository and install the packages needed in requirements.txt and run main.py
## $2e^{-x}(\sin(5x)+x\cos(5x))$
| Dense Configuration | Convolutional Configuration |
| :-------: | :-------: |
| ![densenet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/FullyConnectedNN_5x8.gif) | ![convnet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/ConvolutionalNN_5x8.gif) |
| ![denseresnet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/DenseResNet_5x8.gif) | ![convresnet](https://github.com/navidgolkar/curve-fitting-nn/blob/main/saves/Mish/ConvResNet_5x8.gif) |
