//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include <cstdlib>
#include <cassert>
#include <iostream>

#include "net.h"

class TestOutLayer : public sf::OutputNeuronLayer
{
public:
    inline unsigned long getNeuronsCount() { return this->neurons->size(); }
};

int main(int argc, char const *argv[])
{
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
    
    unsigned long w, h;
    double *res = layer->getOutput(w, h);
    assert(w == 2);
    assert(h == 2);
    assert(res[0] == 7);
    assert(res[1] == 8);
    assert(res[2] == 9);
    assert(res[3] == 6);
    
    TestOutLayer *out = new TestOutLayer();
    out->reserveNeurons(2);
    assert_log(out->getNeuronsCount() == 2, "Neurons reserve fuckup");
    out->reserveNeurons(100);
    assert_log(out->getNeuronsCount() == 100, "Neurons reserve fuckup");
    out->reserveNeurons(3);
    assert_log(out->getNeuronsCount() == 3, "Neurons reserve fuckup");
    out->reserveNeurons(2);
    assert_log(out->getNeuronsCount() == 2, "Neurons reserve fuckup");
    out->reserveNeurons(100);
    assert_log(out->getNeuronsCount() == 100, "Neurons reserve fuckup");
    out->reserveNeurons(3);
    assert_log(out->getNeuronsCount() == 3, "Neurons reserve fuckup");
    out->reserveNeurons(20);
    assert_log(out->getNeuronsCount() == 20, "Neurons reserve fuckup");
    out->reserveNeurons(100);
    assert_log(out->getNeuronsCount() == 100, "Neurons reserve fuckup");
    out->reserveNeurons(3);
    assert_log(out->getNeuronsCount() == 3, "Neurons reserve fuckup");

    std::cout << "All good" << std::endl;
    
    return 0;
}
