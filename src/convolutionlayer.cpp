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
    
}

void sf::ConvolutionLayer::calculateOutput()
{
    
}

double *sf::ConvolutionLayer::getOutput(unsigned long &width, unsigned long &height)
{
    width = this->outputWidth;
    height = this->outputHeight;
    
    return this->output;
}