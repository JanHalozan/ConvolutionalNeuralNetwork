//
//  poolinglayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "poolinglayer.h"

sf::PoolingLayer::PoolingLayer() : Layer()
{
    this->type = sf::kLayerTypePooling;
    this->filterSize = 2;
    this->stride = 2;
}

void sf::PoolingLayer::calculateOutput()
{
    assert_log(this->input != nullptr, "No input");
    
    //Both must be a power of 2
    unsigned long width = this->inputWidth / this->filterSize;
    unsigned long height = this->inputHeight / this->filterSize;
    
    if (!(this->outputWidth == width && this->outputHeight == height))
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputWidth = width;
        this->outputHeight = height;
        auto size = this->outputWidth * this->outputHeight;
        this->output = new double[size];
        this->selectedFilterIndexes = new unsigned char[size];
    }
    
    unsigned long outRow = 0;
    unsigned long outCol = 0;
    
    //Goes first columns then rows
    for (unsigned long row = 0; row < this->inputHeight; row += this->stride)
    {
        for (unsigned long col = 0; col < this->inputWidth; col += this->stride)
        {
            double max = this->input[row * this->inputWidth + col];
            unsigned char selectedMaxIndex = 0;
            
            if (this->input[row * this->inputWidth + col + 1] > max)
            {
                max = this->input[row * this->inputWidth + col + 1];
                selectedMaxIndex = 1;
            }
            
            if (this->input[(row + 1) * this->inputWidth + col] > max)
            {
                max = this->input[(row + 1) * this->inputWidth + col];
                selectedMaxIndex = 2;
            }
            
            if (this->input[(row + 1) * this->inputWidth + col + 1] > max)
            {
                max = this->input[(row + 1) * this->inputWidth + col + 1];
                selectedMaxIndex = 3;
            }
            
            auto index = outRow * this->outputWidth + outCol;
            this->output[index] = max;
            this->selectedFilterIndexes[index] = selectedMaxIndex;
            
            ++outCol;
            outRow += outCol / this->outputWidth;
            outCol %= this->outputWidth;
        }
    }
}

void sf::PoolingLayer::backprop()
{
    
}