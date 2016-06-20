//
//  neuron.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "neuron.h"

double sf::Neuron::learningRate = 0.5;

sf::Neuron::Neuron() : activationType(kNeuronActivationFunctionTypeSig), output(0.0)
{
}

void sf::Neuron::randomizeWeights(long count)
{
#ifndef DEBUG
    srand((unsigned)time(NULL));
#endif
    
    for (long i = 0; i < count; ++i)
    {
        double weight = (rand() / (double)RAND_MAX) - 0.5;
        this->weights.push_back(weight);
    }
}

double sf::Neuron::getWeight(ulong index) const
{
    return this->weights[index];
}

const std::vector<double> sf::Neuron::getWeights() const
{
    return this->weights;
}

void sf::Neuron::loadInput(std::vector<double> input)
{
    this->inputs = input;
    this->inputs.insert(this->inputs.begin(), -1.0); //We add the bias in sigmoid activation
    
    //It's probably a first time load
    if (this->weights.size() == 0)
        this->randomizeWeights(this->inputs.size());
}

void sf::Neuron::calculateOutput()
{
    switch (this->activationType)
    {
        case kNeuronActivationFunctionTypeSig:
        {
            assert_log(this->inputs.size() == this->weights.size(), "Weights and inputs not the same.");
            double sum = 0.0;
            
            for (ulong i = 0; i < this->inputs.size(); ++i)
                sum += this->inputs[i] * this->weights[i];
            
            this->output = 1.0 / (1 + exp(-sum));
        }
            break;
        case kNeuronActivationFunctionTypeConvolution:
        {
            const ulong kernelSide = this->weights.size() - 1;
            double sum = 0.0;
            
            //weights[0] contain the bias
            for (ulong i = 0; i < this->inputs.size(); ++i)
                sum += this->inputs[i + 1] * this->weights[(i % kernelSide) + 1];
            
            sum += this->inputs[0]; //Bias
            
            this->output = 1.0 / (1 + exp(-sum));
        }
            break;
    }
}

double sf::Neuron::getOutput() const
{
    return this->output;
}

void sf::Neuron::recalculateWeights()
{
    switch (this->activationType)
    {
        case kNeuronActivationFunctionTypeSig:
        {
            for (ulong i = 0; i < this->weights.size(); ++i)
                this->weights[i] += this->learningRate * this->gradient * this->inputs[i];
        }
            break;
        case kNeuronActivationFunctionTypeConvolution:
        {
            const ulong kernelSide = this->weights.size() - 1;
            
            for (ulong i = 0; i < this->inputs.size(); ++i)
                this->weights[(i % kernelSide) + 1] += this->inputs[i] * this->gradient;
            
            this->weights[0] += this->gradient * kernelSide * kernelSide;
        }
            break;
    }
}

void sf::Neuron::setGradient(double g)
{
    this->gradient = g;
}

double sf::Neuron::getGradient() const
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