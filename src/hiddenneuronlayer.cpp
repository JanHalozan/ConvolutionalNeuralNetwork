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
    if (this->neurons->empty())
        this->reserveNeurons(this->inputWidth);
    
    if (this->outputWidth != this->inputWidth)
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputWidth = this->inputWidth;
        this->output = new double[this->outputWidth];
    }
    
    unsigned long i = 0;
    std::vector<double> input(this->input, this->input + this->inputWidth);
    
    for (sf::Neuron &n : *this->neurons)
    {
        n.loadInput(input);
        n.calculateOutput();
        this->output[i++] = n.getOutput();
    }
}

void sf::HiddenNeuronLayer::backprop(sf::Layer *, sf::Layer *nextLayer, sf::LayerBackpropInfo *)
{
    for (auto &neuron : *this->neurons)
    {
        double gradientSum = 0.0;
        unsigned long i = 0;
        
        for (auto &nextLayerNeuron : *nextLayer->neurons)
        {
            gradientSum += nextLayerNeuron.getGradient() * nextLayerNeuron.getWeight(i + 1); //i + 1 because index 0 is the threshold
            ++i;
        }
        
        double gradient = neuron.getOutput() * (1.0 - neuron.getOutput()) * gradientSum;
        neuron.setGradient(gradient);
    }
}