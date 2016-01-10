//
//  poolinglayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "poolinglayer.h"

sf::PoolingLayer::PoolingLayer() : Layer(), stride(2), gradients(nullptr)
{
    this->type = sf::kLayerTypePooling;
}

sf::PoolingLayer::~PoolingLayer()
{
    if (this->gradients != nullptr)
        delete[] this->gradients;
}

void sf::PoolingLayer::calculateOutput()
{
    assert_log(this->input != nullptr, "No input");
    
    //TODO: Check that this is valid (must return integer, ...)
    ulong width = this->inputWidth / this->stride;
    ulong height = this->inputHeight / this->stride;
    
    if (!(this->outputWidth == width && this->outputHeight == height))
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputWidth = width;
        this->outputHeight = height;
        this->outputDepth = this->inputDepth;
        const ulong size = this->outputWidth * this->outputHeight * this->outputDepth;
        this->output = new double[size];
        this->selectedFilterIndexes = new unsigned char[size];
    }
    
    const ulong inputSliceSize = this->inputWidth * this->inputHeight;
    const ulong outputSliceSize = this->outputWidth * this->outputHeight;
    ulong outRow = 0;
    ulong outCol = 0;
    ulong outLyr = 0;
    
    for (ulong lyr = 0; lyr < this->inputDepth; ++lyr)
    {
        //Goes first columns then rows
        for (ulong row = 0; row < this->inputHeight; row += this->stride)
        {
            for (ulong col = 0; col < this->inputWidth; col += this->stride)
            {
                //TODO: This thing does a fixed stride = 2 computation. Make it universal.
                const ulong depth = lyr * inputSliceSize;
                ulong index = col + (row * this->inputWidth) + depth;
                double max = this->input[index];
                unsigned char selectedMaxIndex = 0;
                
                index = (col + 1) + (row * this->inputWidth) + depth;
                if (this->input[index] > max)
                {
                    max = this->input[row * this->inputWidth + col + 1];
                    selectedMaxIndex = 1;
                }
                
                index = col + ((row + 1) * this->inputWidth) + depth;
                if (this->input[index] > max)
                {
                    max = this->input[index];
                    selectedMaxIndex = 2;
                }
                
                index = (col + 1) + ((row + 1) * this->inputWidth) + depth;
                if (this->input[index] > max)
                {
                    max = this->input[index];
                    selectedMaxIndex = 3;
                }
                
                index = outCol + (outRow * this->outputWidth) + (outLyr * outputSliceSize);
                this->output[index] = max;
                this->selectedFilterIndexes[index] = selectedMaxIndex;
                
                ++outCol;
                outRow += outCol / this->outputWidth;
                outLyr += (outRow / this->outputHeight) * (outCol / this->outputWidth);
                outCol %= this->outputWidth;
                outRow %= this->outputHeight;
            }
        }
    }
}

void sf::PoolingLayer::backprop(sf::Layer *, sf::Layer *nextLayer, sf::LayerBackpropInfo *)
{
    const auto totalInputSize = this->inputWidth * this->inputHeight * this->inputDepth;
    
    if (this->gradients == nullptr)
        this->gradients = new double[totalInputSize];
    
    const ulong outputSliceSize = this->outputWidth * this->outputHeight;
    const ulong inputSliceSize = this->inputWidth * this->inputHeight;
    
    //Reset the entire gradient map
    memset(this->gradients, 0.0, totalInputSize);
    
    for (ulong lyr = 0; lyr < this->outputDepth; ++lyr)
    {
        for (ulong row = 0; row < this->outputHeight; ++row)
        {
            for (ulong col = 0; col < this->outputWidth; ++col)
            {
                //Convolutional layer only has one neuron
                auto index = nextLayer->type == kLayerTypeConvolutional ? 0 : col + (row * this->outputWidth) + (lyr * outputSliceSize);
                const auto gradient = nextLayer->neurons->at(index).getGradient();
                const auto routeIndex = this->selectedFilterIndexes[index];
                
                //Start index of the gradient frame
                ulong gradientIndex = (col * this->stride) + (row * this->outputWidth * this->stride) + (lyr * inputSliceSize);
                const ulong gradientCol = routeIndex % this->stride;
                const ulong gradientRow = routeIndex / this->stride;
                gradientIndex += gradientCol + (gradientRow * this->inputWidth);
                
                this->gradients[gradientIndex] = gradient;
            }
        }
    }
}

const double *sf::PoolingLayer::getGradients() const
{
    return this->gradients;
}