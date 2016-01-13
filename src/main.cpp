//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "helpers.h"

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <climits>

#include "net.h"

int main(int argc, char const *argv[])
{
    {
        using namespace sf;
        
        ConvolutionLayer *layer = new ConvolutionLayer();
        double input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 5, 3, 1, 6, 7, 6, 9};
        for (int i = 0; i < 18; ++i)
            input[i] /= 10.0;
        
        layer->setOutputFeatureMapsCount(2);
        layer->loadInput(input, 3, 3, 2);
        layer->calculateOutput();
        
        ulong w, h, d;
        double *out = layer->getOutput(w, h, d);
        
        for (ulong z = 0; z < d; ++z)
        {
            for (ulong y = 0; y < h; ++y)
            {
                for (ulong x = 0; x < w; ++x)
                    std::cout << out[x + y * w + z * w * h] << ", ";
                std::cout << std::endl;
            }
            std::cout << std::endl << "----------" << std::endl;
        }
        
        PoolingLayer *lyr = new PoolingLayer();
        
        //Hack together a layer which will be used for routing the gradients
        HiddenNeuronLayer *propLayer = new HiddenNeuronLayer(1);
        propLayer->neurons->at(0).setGradient(2);
        lyr->backprop(nullptr, propLayer);
        
        layer->backprop(nullptr, lyr);
    }
    
    
//    {
//        using namespace sf;
//        
//        PoolingLayer *layer = new PoolingLayer();
//        double input[] = {6, 2, 3, 8, 5, 1, 7, 4, 9, 10, 11, 12, 14, 13, 15, 16};
//        layer->loadInput(input, 4, 2, 2);
//        layer->calculateOutput();
//        unsigned long w, h, d;
//        double *out = layer->getOutput(w, h, d);
//        
//        for (ulong z = 0; z < d; ++z)
//        {
//            for (ulong y = 0; y < h; ++y)
//            {
//                for (ulong x = 0; x < w; ++x)
//                    std::cout << out[x + y * w + z * w * h] << ", ";
//                std::cout << std::endl;
//            }
//            std::cout << std::endl << "----------" << std::endl;
//        }
//        
//        std::cout << std::endl;
//        std::cout << std::endl;
//        
//        //Hack together a layer which will be used for routing the gradients
//        HiddenNeuronLayer *propLayer = new HiddenNeuronLayer(4);
//        propLayer->neurons->at(0).setGradient(2);
//        propLayer->neurons->at(1).setGradient(1);
//        propLayer->neurons->at(2).setGradient(8);
//        propLayer->neurons->at(3).setGradient(3);
//        
//        layer->backprop(nullptr, propLayer);
//        
//        for (int i = 0; i < 2; ++i)
//        {
//            for (int j = 0; j < 2; ++j)
//            {
//                for (int k = 0; k < 4; ++k)
//                    std::cout << layer->getGradientOfNeuron(k + j * 4 + i * 8) << ", ";
//                std::cout << std::endl;
//            }
//            std::cout << std::endl << "----------" << std::endl;
//        }
//        
//        return 0;
//    }
    
    
//    //A really really really simple example of a MLP. Samples 1 & 2 are similar, so are 3 & 4 and 5 & 6. When the net is trained we feed it an example
//    // similar to first two samples and if the answer is class 0 then the MLP is working correctly.
//    {
//        using namespace sf;
//
//        //Size of our input data
//        const unsigned long inputWidth = 3;
//        const unsigned long inputHeight = 1;
//
//        //A bunch of samples. The 1 & 2 are similar so are 3 & 4 and 5 & 6.
//        double sample1[] = {1.0, 0.2, 0.1};     //Cow
//        double sample2[] = {0.8, 0.1, 0.25};    //Cow
//        double sample3[] = {0.2, 0.95, 0.1};    //Chicken
//        double sample4[] = {0.11, 0.9, 0.13};   //Chicken
//        double sample5[] = {0.0, 0.2, 0.91};    //Car
//        double sample6[] = {0.21, 0.12, 1.0};   //Car
//
//
//        //A new network with the given data width and height
//        Net *net = new Net(inputWidth, inputHeight);
//        net->addLayer(new HiddenNeuronLayer(4)); //A hidden neural layer with 4 neurons
//        net->addLayer(new HiddenNeuronLayer(4)); //A hidden neural layer with 4 neurons
//        net->addLayer(new OutputNeuronLayer()); //Finish it off by adding an output layer
//
//        //Add all the samples with their corresponding labels
//        net->addTrainingSample(sample1, "cow");
//        net->addTrainingSample(sample2, "cow");
//        net->addTrainingSample(sample3, "chicken");
//        net->addTrainingSample(sample4, "chicken");
//        net->addTrainingSample(sample5, "car");
//        net->addTrainingSample(sample6, "car");
//
//        //And now we play the waiting game
//        net->train();
//
//        //This example is similar to "chicken" so we expect the chicken probability to be close to 1 and car and cow to be close to 0
//        double example[] = {0.0, 0.8, 0.1};
//        auto output = net->classifySample(example);
//
//        //Let's see what we get
//        for (auto &tuple : output)
//            std::cout << std::get<1>(tuple) << ": " << std::get<0>(tuple) << std::endl;
//
//        std::cout << std::endl;
//
//        return 0;
//    }
    
//    {
//        
//        using namespace sf;
//        
//        ConvolutionLayer *layer = new ConvolutionLayer();
//        double input[] = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 5, 4, 5, 1, 2, 3};//2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 5, 4, 5, 1, 1, 5, 4, 5, 1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 5, 4, 5, 1, 1, 5, 4, 5, 1};
//        for (unsigned long i = 0; i < sizeof(input) / sizeof(double); ++i)
//            input[i] /= 10.0;
//        
//        layer->loadInput(input, 3, 3, 2);
//        layer->setOutputFeatureMapsCount(2);
//        layer->calculateOutput();
//        
//        unsigned long w, h, d;
//        double *out = layer->getOutput(w, h, d);
//        
//        for (ulong z = 0; z < d; ++z)
//        {
//            for (ulong y = 0; y < h; ++y)
//            {
//                for (ulong x = 0; x < w; ++x)
//                    std::cout << out[x + y * w + z * w * h] << ", ";
//                std::cout << std::endl;
//            }
//            std::cout << std::endl << "----------" << std::endl;
//        }
//        
//        for (unsigned long long i = 0; i < w * h * d; ++i)
//            std::cout << out[i] << ", ";
//        
//        std::cout << std::endl;
//    }
//    
//    //This example was used for debugging backprop
//    {
//        using namespace sf;
//        
//        double sample1[] = {0, 0};
//        double sample2[] = {0, 1};
//        double sample3[] = {1, 0};
//        double sample4[] = {1, 1};
//        
//        Net *net = new Net(2, 1);
//        
//        net->addTrainingSample(sample1, 0);
////        net->addTrainingSample(sample2, 1);
////        net->addTrainingSample(sample3, 1);
////        net->addTrainingSample(sample4, 0);
//        
//        net->addLayer(new HiddenNeuronLayer(2));
//        net->addLayer(new OutputNeuronLayer());
//        net->layers.back()->reserveNeurons(1);
//        
//        net->train();
//        
//        return 0;
//    }
//    
//    
//    
//    //Below is a bunch of "unit tests". These are just for testing purposes that things behave the way they should.
//    sf::Neuron::learningRate = 0.5;
//    
//    double sum = (0.1 * -1) + (-0.1 * 1) + (0.5 * 2) + (0.2 * 3);
//    double prefOut = 1.0 / (1 + exp(-sum));
//    sf::Neuron n;
//    n.activationType = sf::kNeuronActivationFunctionTypeSig;
//    std::vector<double> input = {1, 2, 3};
//    n.weights = {0.1, -0.1, 0.5, 0.2};
//    n.loadInput(input);
//    n.calculateOutput();
//    assert_log(n.output == prefOut, "Neuron output fail");
//    n.gradient = 0.25;
//    n.recalculateWeights();
//    assert_log(n.weights[0] == 0.1 + (0.5 * 0.25 * -1), "Recalculate weights fail");
//    assert_log(n.weights[1] == -0.1 + (0.5 * 0.25 * 1), "Recalculate weights fail");
//    assert_log(n.weights[2] == 0.5 + (0.5 * 0.25 * 2), "Recalculate weights fail");
//    assert_log(n.weights[3] == 0.2 + (0.5 * 0.25 * 3), "Recalculate weights fail");
//    
//    sf::PoolingLayer *layer = new sf::PoolingLayer();
//    double data[] = {
//        4, 5, 4, 5,
//        6, 7, 8, 7,
//        2, 9, 6, 1,
//        3, 6, 5, 4
//    };
//    
//    layer->loadInput(data, 4, 4);
//    layer->calculateOutput();
//    
//    unsigned long w, h, d;
//    double *res = layer->getOutput(w, h, d);
//    assert(w == 2);
//    assert(h == 2);
//    assert(res[0] == 7);
//    assert(res[1] == 8);
//    assert(res[2] == 9);
//    assert(res[3] == 6);
//
//    std::cout << "All good" << std::endl;
    
    return 0;
}
