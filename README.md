##ConvolutionalNeuralNetwork

A general purpose convolutional neural network for classification.

**Under construction.** This network it currently under development and expected to be finished soon. I will also write a blog post about every single step of the way. Stay tuned.

####Usage

The network has a layered design which allows different configurations of how many layers and what layer types should be stacked together. The usage should be similar to:

```
+-------------------+---------------+-----+--------------+-----+--------------+
| Convolution layer | Pooling layer | ... | Hidden layer | ... | Output layer |
+-------------------+---------------+-----+--------------+-----+--------------+
```

The convolution & pooling layer combinations are optional and if one uses just some hidden layers and the output layer the network becomes a regular multilayer perceptron. The pooling layer itself is also optional but practice has shown that the network gives better results when pooling layers are used (spatial invariance).

Types of layers:

- Convolution. The convolution layer will perform the main feature extraction for the provided sample. Each layer must have a pre-defined number of different feature extractor (the depth).
- Pooling. The pooling layer performs a downsampling of the input by the factor of `filterSize` (defaults to 2) and with a stride provided in `stride` (defaults to 2). So the default downsampling is equal to 2x2.
- Hidden. The hidden layer is a layer of a regular multi layer perceptron. It's only difference from the output layer is the way it computes its backpropagation gradients.
- Output layer. The final layer. There are no hyperparameters to specify, just make sure you end the network with an output layer.

**Code**
Here's what an example might look like (will probably change as I get more and more stuff done):

```c++

const unsigned long inputWidth = 32; //The dimension of your input. This example uses a 32x32 "pixels" input.
const unsigned long inputHeight = 32; //You can also specify a 1D input by setting the height equal to 1.
Net *net = new Net(inputWidth, inputHeight);

// net->addLayer(new ConvolutionLayer()); WORK IN PROGRESS. I still have to decide which hyperparameters to specify here.
// net->addLayer(new PoolingLayer()); WORK IN PROGRESS.

// net->addLayer(new HiddenNeuronLayer()); WORK IN PROGRESS.
net->addLayer(new OutputNeuronLayer());

for (auto sample : mySamplesVector)
{
    net->addTrainingSample(sample.data, sample.classNumber);
}

net->train(); //And now we play the waiting game.

double *output = net->classifySample(testSample.data);

std::cout << "Classification restults: ";
for (unsigned short i = 0; i < differentClassesCount; ++i)
    std::cout << output[i] << ", ";

```

####Building it

You can open it using Xcode or just `make build`. To toggle debug mode (exposes all private properties, ...) just update the `#define DEBUG` flag in the `helpers.h`.
