//
//  neuron.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef neuron_h
#define neuron_h

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "helpers.h"

namespace sf
{
    class Neuron;
}

class sf::Neuron
{
private:
    std::vector<double> weights;
    std::vector<double> inputs;

    double output;
    double gradient;
    
public:
    static double learningRate;
    
    Neuron();
    
    void randomizeWeights();
    void loadInput(std::vector<double> input);
    void calculateOutput();
    double getOutput();
    
    void backpop(std::vector<double> values);
    double getGradient();
};

#endif /* neuron_h */
