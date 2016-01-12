##ConvolutionalNeuralNetwork

A general purpose convolutional neural network.

**Under construction. This network it currently under development and expected to be finished soon.**  
When finished I will write a blog post on how to write the thing from scratch yourself. Feel free to _star_ the repo.

###About

This project is a simple to use general purpose convolutional neural network framework. It features several types of layers which can be linked together as they are needed. Written in C++ which allows it to run blazingly fast (not just yet) and stay extremely portable (it has no dependencies).

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

//Size of our input data
const unsigned long inputWidth = 3;
const unsigned long inputHeight = 1;

//A bunch of samples. The 1 & 2 are similar so are 3 & 4 and 5 & 6.
double sample1[] = {1.0, 0.2, 0.1};     //Cow
double sample2[] = {0.8, 0.1, 0.25};    //Cow
double sample3[] = {0.2, 0.95, 0.1};    //Chicken
double sample4[] = {0.11, 0.9, 0.13};   //Chicken
double sample5[] = {0.0, 0.2, 0.91};    //Car
double sample6[] = {0.21, 0.12, 1.0};   //Car


//A new network with the given data width and height
Net *net = new Net(inputWidth, inputHeight);
net->addLayer(new HiddenNeuronLayer(4)); //A hidden neural layer with 4 neurons
net->addLayer(new HiddenNeuronLayer(4)); //A hidden neural layer with 4 neurons
net->addLayer(new OutputNeuronLayer()); //Finish it off by adding an output layer

//Add all the samples with their corresponding labels
net->addTrainingSample(sample1, "cow");
net->addTrainingSample(sample2, "cow");
net->addTrainingSample(sample3, "chicken");
net->addTrainingSample(sample4, "chicken");
net->addTrainingSample(sample5, "car");
net->addTrainingSample(sample6, "car");

//And now we play the waiting game
net->train();

//This example is similar to "chicken" so we expect the chicken probability to be close to 1 and car and cow to be close to 0
double example[] = {0.0, 0.8, 0.1};
auto output = net->classifySample(example);

//Let's see what we get
for (auto &tuple : output)
    std::cout << std::get<1>(tuple) << ": " << std::get<0>(tuple) << std::endl;

std::cout << std::endl;
```

####Building it

You can open it using Xcode or just `make build`. To toggle debug mode (exposes all private properties, ...) just update the `#define DEBUG` flag in the `helpers.h`.

###TODO

Things to come (in order):  
- [ ] Finish `PoolingLayer` backpropagation
- [ ] Finish `ConvolutionLayer` backpropagation
- [ ] A few tweaks here and there
- [ ] Release alpha version
- [ ] Merge `OutputNeuronLayer` and `HiddenNeuronLayer`
- [ ] Refactor to C++14 (no raw pointers, ...)
- [ ] Finish various TODOs (code comments)
- [ ] Release beta version  
After this point I don't have a concrete plan of what to do next. Here are some things I'm considering:
- [ ] Speed it up
- [ ] CUDA support
- [ ] Add tests :>
- [ ] ???
