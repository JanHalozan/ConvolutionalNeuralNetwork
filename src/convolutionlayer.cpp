//
//  convolutionlayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "convolutionlayer.h"

sf::ConvolutionLayer::ConvolutionLayer(unsigned long depth) : Layer(), stride(1), kernelSide(3), zeroPaddingSize(0)
{
    this->type = kLayerTypeConvolutional;
    this->reserveNeurons(depth);
}

void sf::ConvolutionLayer::calculateOutput()
{
    assert_log(ceil((this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride) == (this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride, "Invalid hyper parameters (width)");
    assert_log(ceil((this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride) == (this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride, "Invalid hyper parameters (height)");
    
    unsigned long oWidth = (this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride;
    unsigned long oHeight = (this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride;
    
    if (!(this->outputWidth == oWidth && this->outputHeight == oHeight))
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputDepth = this->inputDepth;
        this->outputWidth = oWidth;
        this->outputHeight = oHeight;
        this->output = new double[this->outputWidth * this->outputHeight];
    }
    
    const unsigned long sliceSize = this->inputWidth * this->inputHeight;
    const unsigned long outputSliceSize = this->outputWidth * this->outputHeight;
    unsigned long outRow = 0;
    unsigned long outCol = 0;
    
    for (unsigned long row = 0; row < this->inputHeight; ++row)
    {
        for (unsigned long col = 0; col < this->inputWidth; ++col)
        {
            for (unsigned long lyr = 0; lyr < this->inputDepth; ++lyr)
            {
                std::vector<double> input(this->kernelSide * this->kernelSide);
                for (unsigned long y = 0; y < this->kernelSide; ++y)
                {
                    for (unsigned long x = 0; x < this->kernelSide; ++x)
                    {
                        double val = this->input[(col + x) + ((row + y) * this->inputWidth) + (lyr * sliceSize)];
                        input.push_back(val);
                    }
                }
                
                unsigned short depth = 0;
            
                for (auto &neuron : *this->neurons)
                {
                    neuron.loadInput(input);
                    neuron.calculateOutput();
                    
                    this->output[outCol + (outRow * this->outputWidth) + (outputSliceSize * depth)] = neuron.getOutput();
                    
                    ++outCol;
                    outRow += outCol / this->outputWidth;
                    outCol %= this->outputWidth;
                    ++depth;
                }
            }
        }
    }
}

void sf::ConvolutionLayer::backprop(sf::Layer *previousLayer, sf::Layer *nextLayer, sf::LayerBackpropInfo *info)
{
    
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

void sf::ConvolutionLayer::setZeroPaddingSize(unsigned char size)
{
    this->zeroPaddingSize = size;
}

unsigned char sf::ConvolutionLayer::getZeroPaddingSize() const
{
    return this->zeroPaddingSize;
}