//
//  convolutionlayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "convolutionlayer.h"

sf::ConvolutionLayer::ConvolutionLayer() : Layer(), stride(1), kernelSide(3), zeroPaddingSize(0)
{
    this->type = kLayerTypeConvolutional;
}

void sf::ConvolutionLayer::calculateOutput()
{
    assert_log(ceil((this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride) == (this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride, "Invalid hyper parameters (width)");
    assert_log(ceil((this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride) == (this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride, "Invalid hyper parameters (height)");
    
    ulong oWidth = (this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride;
    ulong oHeight = (this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride;
    
    if (!(this->outputWidth == oWidth && this->outputHeight == oHeight))
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputDepth = this->inputDepth;
        this->outputWidth = oWidth;
        this->outputHeight = oHeight;
        this->output = new double[this->outputWidth * this->outputHeight];
    }
    
    const ulong sliceSize = this->inputWidth * this->inputHeight;
    const ulong outputSliceSize = this->outputWidth * this->outputHeight;
    ulong outRow = 0;
    ulong outCol = 0;
    
    for (long long row = -this->zeroPaddingSize; row < (signed)(this->inputHeight + this->zeroPaddingSize); row += this->stride)
    {
        for (long long col = -this->zeroPaddingSize; col < (signed)(this->inputWidth + this->zeroPaddingSize); col += this->stride)
        {
            unsigned short depth = 0;
            for (auto &neuron : *this->neurons)
            {
                std::vector<double> input(this->kernelSide * this->kernelSide * this->inputDepth);
                
                for (ulong lyr = 0; lyr < this->inputDepth; ++lyr)
                {
                    for (ulong y = 0; y < this->kernelSide; ++y)
                    {
                        for (ulong x = 0; x < this->kernelSide; ++x)
                        {
                            if (row < 0 || row >= (signed)this->inputHeight || col < 0 || col >= (signed)this->inputWidth)
                            {
                                input.push_back(0.0);
                                continue;
                            }
                            
                            double val = this->input[(col + x) + ((row + y) * this->inputWidth) + (lyr * sliceSize)];
                            input.push_back(val);
                        }
                    }
                }
                
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

void sf::ConvolutionLayer::setOutputFeatureMapsCount(ulong count)
{
    this->reserveNeurons(count);
    this->outputDepth = count;
}

ulong sf::ConvolutionLayer::getOutputFeatureMapsCount()
{
    return this->outputDepth;
}