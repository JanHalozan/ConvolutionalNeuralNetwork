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
}

void sf::OutputNeuronLayer::calculateOutput()
{
    
}

void sf::OutputNeuronLayer::backprop()
{
    std::vector<double> errors;
    
}