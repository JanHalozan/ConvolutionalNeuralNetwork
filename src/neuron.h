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
    
    typedef enum {
        kNeuronActivationFunctionTypeSig = 1,
        kNeuronActivationFunctionTypeConvolution = 2
    } NeuronActivationFunctionType;
}

class sf::Neuron
{
private:
    sf::NeuronActivationFunctionType activationType;
    
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
    
    void backprop(std::vector<double> values);
    double getGradient();
    
    void setActivationFunctionType(sf::NeuronActivationFunctionType t);
    sf::NeuronActivationFunctionType getActivationFunctionType();
};

#endif /* neuron_h */
