//
//  outputneuronlayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "outputneuronlayer.h"

sf::OutputNeuronLayer::OutputNeuronLayer() : sf::Layer()
{
    this->type = kLayerTypeOutputNeuron;
    this->outputHeight = 1;
}

void sf::OutputNeuronLayer::setBackpropTargetNeuron(ulong index)
{
    this->backpropTargetNeuron = index;
}

void sf::OutputNeuronLayer::calculateOutput()
{
    if (this->outputWidth != this->inputWidth)
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputWidth = (ulong)this->neurons->size();
        this->outputHeight = 1;
        this->outputDepth = 1;
        this->output = new double[this->outputWidth];
    }
    
    ulong i = 0;
    const ulong inputSize = this->inputWidth * this->inputHeight * this->inputDepth;
    std::vector<double> input(this->input, this->input + inputSize);
    
    for (sf::Neuron &n : *this->neurons)
    {
        n.loadInput(input);
        n.calculateOutput();
        this->output[i++] = n.getOutput();
    }
}

void sf::OutputNeuronLayer::backprop(sf::Layer *, sf::Layer *)
{
    //Calculate the gradients and recalculate the output for each neuron in the output layer
    ulong i = 0;
    for (auto &n : *this->neurons)
    {
        double desiredOutput = this->backpropTargetNeuron == i ? 1.0 : 0.0;
        double actualOutput = n.getOutput();
        double gradient = actualOutput * (1.0 - actualOutput) * (desiredOutput - actualOutput);
        n.setGradient(gradient);
        
        ++i;
    }
}