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
    this->reserveGradientItems(1);
}

void sf::Neuron::randomizeWeights(const long count)
{
#ifndef DEBUG
    srand((unsigned)time(NULL));
#endif
    
    for (long i = 0; i < count; ++i)
    {
        const double weight = (rand() / (double)RAND_MAX) - 0.5;
        this->weights.push_back(weight);
    }
}

double sf::Neuron::getWeight(const ulong index) const
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
            
            this->output = 1.0 / (1.0 + exp(-sum));
        }
            break;
        case kNeuronActivationFunctionTypeConvolution:
        {
            double sum = 0.0;
            
            //weights[0] contain the bias
            for (ulong i = 1; i < this->inputs.size() + 1; ++i)
                sum += this->inputs[i] * this->weights[i];
            
            sum += this->inputs[0]; //Bias
            
            this->output = 1.0 / (1.0 + exp(-sum));
        }
            break;
    }
}

double sf::Neuron::getOutput() const
{
    return this->output;
}

void sf::Neuron::recalculateWeights(const ulong gradientIndex)
{
    switch (this->activationType)
    {
        case kNeuronActivationFunctionTypeSig:
        {
            for (ulong i = 0; i < this->weights.size(); ++i)
                this->weights[i] += this->learningRate * this->getGradient() * this->inputs[i];
        }
            break;
        case kNeuronActivationFunctionTypeConvolution:
        {
            
//            for (ulong i = 0; i < this->weights.size(); ++i)
//                this->weights[i] += this->learningRate * this->getGradient(gradientIndex) * this->inputs[i];
            
            
//            const ulong kernelSide = this->weights.size() - 1;
//            
//            for (ulong i = 0; i < this->inputs.size(); ++i)
//                this->weights[(i % kernelSide) + 1] += this->inputs[i] * this->getGradient(i);
//            
//            this->weights[0] += this->getGradient() * kernelSide * kernelSide;
        }
            break;
    }
}

void sf::Neuron::reserveGradientItems(const ulong count)
{
    if (this->gradients.size() != count)
        this->gradients.resize(count);
}

void sf::Neuron::setGradient(const double g, const ulong index)
{
    this->gradients[index] = g;
}

double sf::Neuron::getGradient(const ulong index) const
{
    return this->gradients[index];
}

void sf::Neuron::setActivationFunctionType(const sf::NeuronActivationFunctionType t)
{
    this->activationType = t;
    
    switch (t)
    {
        case kNeuronActivationFunctionTypeSig:
            if (this->gradients.size() != 1)
            {
                this->gradients.resize(1);
            }
            break;
        case kNeuronActivationFunctionTypeConvolution:
            break;
    }
}

sf::NeuronActivationFunctionType sf::Neuron::getActivationFunctionType() const
{
    return this->activationType;
}