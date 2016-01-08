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
//    //A really really really simple example of a MLP. Samples 1 & 2 are similar, so are 3 & 4 and 5 & 6. When the net is trained we feed it an example
//    // similar to first two samples and if the answer is class 0 then the MLP is working correctly.
//    {
//        using namespace sf;
//        
//        double sample1[] = {1.0, 0.2, 0.1};
//        double sample2[] = {0.8, 0.1, 0.25};
//        double sample3[] = {0.2, 0.95, 0.1};
//        double sample4[] = {0.11, 0.9, 0.13};
//        double sample5[] = {0.0, 0.2, 0.91};
//        double sample6[] = {0.21, 0.12, 1.0};
//        
//        Net *net = new Net(3, 1);
//        net->addLayer(new HiddenNeuronLayer(4));
//        net->addLayer(new HiddenNeuronLayer(4));
//        net->addLayer(new OutputNeuronLayer());
//        
//        net->addTrainingSample(sample1, 0);
//        net->addTrainingSample(sample2, 0);
//        net->addTrainingSample(sample3, 1);
//        net->addTrainingSample(sample4, 1);
//        net->addTrainingSample(sample5, 2);
//        net->addTrainingSample(sample6, 2);
//        
//        net->train();
//        
//        double example[] = {1.0, 0.2, 0.11};
//        double *output = net->classifySample(example);
//        for (int i = 0; i < 3; ++i)
//            std::cout << output[i] << ", ";
//        
//        std::cout << std::endl;
//        
//        return 0;
//    }
    
    {
        
        using namespace sf;
        
        ConvolutionLayer *layer = new ConvolutionLayer();
        double input[] = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 5, 4, 5, 1, 2, 3};//2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 5, 4, 5, 1, 1, 5, 4, 5, 1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 1, 5, 4, 5, 1, 1, 5, 4, 5, 1};
        for (unsigned long i = 0; i < sizeof(input) / sizeof(double); ++i)
            input[i] /= 10.0;
        
        layer->loadInput(input, 3, 3, 2);
        layer->setOutputFeatureMapsCount(2);
        layer->calculateOutput();
        
        unsigned long w, h, d;
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
        
        for (auto i = 0; i < w * h * d; ++i)
            std::cout << out[i] << ", ";
        
        std::cout << std::endl;
    }
    
//    This example was used for debugging backprop
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
    
    
    
    //Below is a bunch of "unit tests". These are just for testing purposes that things behave the way they should.
    sf::Neuron::learningRate = 0.5;
    
    double sum = (0.1 * -1) + (-0.1 * 1) + (0.5 * 2) + (0.2 * 3);
    double prefOut = 1.0 / (1 + exp(-sum));
    sf::Neuron n;
    n.activationType = sf::kNeuronActivationFunctionTypeSig;
    std::vector<double> input = {1, 2, 3};
    n.weights = {0.1, -0.1, 0.5, 0.2};
    n.loadInput(input);
    n.calculateOutput();
    assert_log(n.output == prefOut, "Neuron output fail");
    n.gradient = 0.25;
    n.recalculateWeights();
    assert_log(n.weights[0] == 0.1 + (0.5 * 0.25 * -1), "Recalculate weights fail");
    assert_log(n.weights[1] == -0.1 + (0.5 * 0.25 * 1), "Recalculate weights fail");
    assert_log(n.weights[2] == 0.5 + (0.5 * 0.25 * 2), "Recalculate weights fail");
    assert_log(n.weights[3] == 0.2 + (0.5 * 0.25 * 3), "Recalculate weights fail");
    
    sf::PoolingLayer *layer = new sf::PoolingLayer();
    double data[] = {
        4, 5, 4, 5,
        6, 7, 8, 7,
        2, 9, 6, 1,
        3, 6, 5, 4
    };
    
    layer->loadInput(data, 4, 4);
    layer->calculateOutput();
    
    unsigned long w, h, d;
    double *res = layer->getOutput(w, h, d);
    assert(w == 2);
    assert(h == 2);
    assert(res[0] == 7);
    assert(res[1] == 8);
    assert(res[2] == 9);
    assert(res[3] == 6);
    
    auto out = new sf::OutputNeuronLayer;
    out->reserveNeurons(2);
    assert_log(out->neurons->size() == 2, "Neurons reserve fuckup");
    out->reserveNeurons(100);
    assert_log(out->neurons->size() == 100, "Neurons reserve fuckup");
    out->reserveNeurons(3);
    assert_log(out->neurons->size() == 3, "Neurons reserve fuckup");
    out->reserveNeurons(2);
    assert_log(out->neurons->size() == 2, "Neurons reserve fuckup");
    out->reserveNeurons(100);
    assert_log(out->neurons->size() == 100, "Neurons reserve fuckup");
    out->reserveNeurons(3);
    assert_log(out->neurons->size() == 3, "Neurons reserve fuckup");
    out->reserveNeurons(20);
    assert_log(out->neurons->size() == 20, "Neurons reserve fuckup");
    out->reserveNeurons(100);
    assert_log(out->neurons->size() == 100, "Neurons reserve fuckup");
    out->reserveNeurons(3);
    assert_log(out->neurons->size() == 3, "Neurons reserve fuckup");
    
    //33614 564950498 1097816499 1969887316
    std::cout << RAND_MAX << std::endl; //RAND_MAX = 2147483647
    const auto samples = 2;
    srand(2);
    auto net = new sf::Net(3, 1);
    net->addLayer(new sf::OutputNeuronLayer());
    double sample[] = {1, 2, 8};
//    net->addTrainingSample(sample, 0);
    net->layers.front()->reserveNeurons(samples);
    double *output = net->classifySample(sample);
    assert_log(net->layers.front()->neurons->front().weights[0] - (-0.499984) < std::numeric_limits<double>::epsilon(), "Weights fuckup");
    assert_log(output[0] - 0.97401439427008673 < std::numeric_limits<double>::epsilon(), "Output fuckup");
    
    for (int i = 0; i < samples; ++i)
        std::cout << output[i] << ", ";
    
    net->layers.front()->neurons->front().gradient = 0.23;
    net->layers.front()->neurons->front().recalculateWeights();
    assert_log(net->layers.front()->neurons->front().weights[0] - (-0.499984 + (-0.499984 * 0.23 * 1)) < std::numeric_limits<double>::epsilon(), "Weights fuckup");

    std::cout << "All good" << std::endl;
    
    return 0;
}
