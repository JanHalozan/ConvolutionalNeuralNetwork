//
//  convolutionlayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "convolutionlayer.h"

sf::ConvolutionLayer::ConvolutionLayer(unsigned long depth) : Layer(), stride(1), kernelSide(3), useZeroPadding(false)
{
    this->type = kLayerTypeConvolutional;
    this->reserveNeurons(depth);
}

void sf::ConvolutionLayer::calculateOutput()
{
    
}

void sf::ConvolutionLayer::backprop(sf::Layer *previousLayer, sf::Layer *nextLayer, sf::LayerBackpropInfo *info)
{
    
}

double sf::ConvolutionLayer::convolve(double *input, double *kernel)
{
    double result = 0.0;
    
    
    
    return result;
}

void sf::ConvolutionLayer::setStride(unsigned short stride)
{
    this->stride = stride;
}

unsigned short sf::ConvolutionLayer::getStride() const
{
    return this->stride;
}

void sf::ConvolutionLayer::setKernelSideSize(unsigned short size)
{
    this->kernelSide = size;
}

unsigned short sf::ConvolutionLayer::getKernelSideSize() const
{
    return this->kernelSide;
}

void sf::ConvolutionLayer::setUseZeroPadding(bool use)
{
    this->useZeroPadding = use;
}

bool sf::ConvolutionLayer::getUseZeroPadding() const
{
    return this->useZeroPadding;
}