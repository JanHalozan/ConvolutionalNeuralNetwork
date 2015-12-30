//
//  neuron.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "neuron.h"

double sf::Neuron::learningRate = 0.5;

sf::Neuron::Neuron() : output(0.0)
{
}

void sf::Neuron::randomizeWeights()
{
    srand((unsigned)time(NULL));
    
    for (unsigned long i = 0; i < this->inputs.size(); ++i)
    {
        double weight = (rand() / (double)RAND_MAX) - 0.5;
        this->weights.push_back(weight);
    }
}

void sf::Neuron::loadInput(std::vector<double> input)
{
    this->inputs = input;
    this->inputs.insert(this->inputs.begin(), -1);
    
    //It's probably a first time load
    if (this->weights.size() == 0)
        this->randomizeWeights();
}

void sf::Neuron::calculateOutput()
{
    assert_log(this->inputs.size() == this->weights.size(), "Weights and inputs not the same.");
    
    switch (this->activationType)
    {
        case kNeuronActivationFunctionTypeSig:
        {
            double sum = 0.0;
            
            for (unsigned long i = 0; i < this->inputs.size(); ++i)
            {
                double tmp = this->inputs[i] * this->weights[i];
                sum += tmp;
            }
            
            this->output = 1.0 / (1 + exp(-sum));
        }
            break;
        case kNeuronActivationFunctionTypeConvolution:
        {
            
        }
            break;
    }
}

double sf::Neuron::getOutput()
{
    return this->output;
}

void sf::Neuron::backprop(std::vector<double> values)
{
    
}

double sf::Neuron::getGradient()
{
    return this->gradient;
}

void sf::Neuron::setActivationFunctionType(sf::NeuronActivationFunctionType t)
{
    this->activationType = t;
}

sf::NeuronActivationFunctionType sf::Neuron::getActivationFunctionType()
{
    return this->activationType;
}