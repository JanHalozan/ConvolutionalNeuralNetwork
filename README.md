##ConvolutionalNeuralNetwork

A general purpose convolutional neural network.

**Under construction. This network it currently under development and expected to be finished soon.**  
When finished I will write a blog post on how to write the thing from scratch yourself. Feel free to _star_ the repo.

###About

This project is a simple to use general purpose convolutional neural network framework. It features several types of layers which can be linked together as they are needed. Written in C++ which allows it to run blazingly fast and stay extremely portable (it has no dependencies).

###Usage

The network has a layered design which allows different configurations of how many layers and what layer types should be stacked together.

There are four types of layers:

- **Convolution**. The convolution layer will perform the main feature extraction for the provided sample.
- **Pooling**. The pooling layer performs a downsampling of the input by the factor of `filterSize` (defaults to 2) and with a stride provided in `stride` (defaults to 2). So the default downsampling is equal to 2x2.
- **Hidden**. The hidden layer is a layer of a regular multi layer perceptron. It's only difference from the output layer is the way it computes its backpropagation gradients.
- **Output layer**. The final layer. There are no hyperparameters to specify, just make sure you end the network with an output layer.

A fancy ASCII art of the network structure:

```
+-------------------+---------------+-----+--------------+-----+--------------+
| Convolution layer | Pooling layer | ... | Hidden layer | ... | Output layer |
+-------------------+---------------+-----+--------------+-----+--------------+
```

The convolution & pooling layer combinations are optional and there are no restrictions on how many there can be. If no conv & pooling layers are present the netowrk behaves line a regular multilayer perceptron. While the pooling layer itself is also optional it is recommended since the network yields better results when used (spatial invariance).

#####Code

Here's what an example might look like (so far only the MLP part is working):

```c++
using namespace sf;

const unsigned long inputWidth = 3;
const unsigned long inputHeight = 1;

//A bunch of samples. The 1 & 2 are similar so are 3 & 4 and 5 & 6. 
double sample1[] = {1.0, 0.2, 0.1};
double sample2[] = {0.8, 0.1, 0.25};
double sample3[] = {0.2, 0.95, 0.1};
double sample4[] = {0.11, 0.9, 0.13};
double sample5[] = {0.0, 0.2, 0.91};
double sample6[] = {0.21, 0.12, 1.0};

//A new network with the given data width and height
Net *net = new Net(inputWidth, inputHeight);
net->addLayer(new HiddenNeuronLayer(4)); //A hidden neural layer with 4 neurons
net->addLayer(new HiddenNeuronLayer(4)); //A hidden neural layer with 4 neurons
net->addLayer(new OutputNeuronLayer()); //An output layer

//Add the samples
net->addTrainingSample(sample1, 0); //Sample 1 & 2 belong to class 0 (first index of the output array)
net->addTrainingSample(sample2, 0);
net->addTrainingSample(sample3, 1); //Sample 3 & 5 belong to class 2 (second index of the output array)
net->addTrainingSample(sample4, 1);
net->addTrainingSample(sample5, 2); //Sample 5 & 6 belong to class 3 (third index of the output array)
net->addTrainingSample(sample6, 2);

//And now we play the waiting game
net->train();

//This example input is very similar to the sample 1 and 2 so we expect our output to have a value
//close to 1 for class 0 and a value close to 0 for other classes.

double example[] = {1.0, 0.2, 0.11};
double *output = net->classifySample(example);
for (int i = 0; i < 3; ++i)
    std::cout << output[i] << ", ";

```

####Building it

You can open it using Xcode or just `make build`. To toggle debug mode (exposes all private properties, ...) just update the `#define DEBUG` flag in the `helpers.h`.
