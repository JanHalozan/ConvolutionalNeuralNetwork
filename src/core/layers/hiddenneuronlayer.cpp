//
//  hiddenneuronlayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 02/01/16.
//  Copyright © 2016 JanHalozan. All rights reserved.
//

#include "hiddenneuronlayer.h"

sf::HiddenNeuronLayer::HiddenNeuronLayer() : sf::Layer()
{
    this->type = kLayerTypeHiddenNeuron;
}

void sf::HiddenNeuronLayer::calculateOutput()
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

void sf::HiddenNeuronLayer::backprop(sf::Layer *, sf::Layer *nextLayer)
{
    ulong i = 0;
    
    for (auto &neuron : *this->neurons)
    {
        double gradientSum = 0.0;
        
        for (const auto &nextLayerNeuron : nextLayer->getNeurons())
            gradientSum += nextLayerNeuron.getGradient() * nextLayerNeuron.getWeight(i + 1); //i + 1 because index 0 is the bias
        
        const double gradient = neuron.getOutput() * (1.0 - neuron.getOutput()) * gradientSum;
        neuron.setGradient(gradient);
        ++i;
    }
}

double sf::HiddenNeuronLayer::getGradientOfNeuron(ulong x, ulong y, ulong z) const
{
    return this->neurons->at(x + y + z).getGradient();
}
