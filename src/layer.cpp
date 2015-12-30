//
//  layer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "layer.h"

sf::Layer::Layer()
{
    this->output = nullptr;
    this->input = nullptr;
}

sf::LayerType sf::Layer::getType()
{
    return this->type;
}

void sf::Layer::loadInput(double *input, unsigned long width, unsigned long height)
{
    this->inputWidth = width;
    this->inputHeight = height;
    this->input = input;
}

double *sf::Layer::getOutput(unsigned long &width, unsigned long &height)
{
    width = this->outputWidth;
    height = this->outputHeight;
    
    return this->output;
}