//
//  convolutionlayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "convolutionlayer.h"

sf::ConvolutionLayer::ConvolutionLayer() : Layer()
{
    this->type = kLayerTypeConvolutional;
}

void sf::ConvolutionLayer::calculateOutput()
{
    
}

void sf::ConvolutionLayer::backprop(sf::Layer *previousLayer, sf::Layer *nextLayer, sf::LayerBackpropInfo *info)
{
    
}