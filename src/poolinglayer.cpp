//
//  poolinglayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "poolinglayer.h"

sf::PoolingLayer::PoolingLayer() : Layer(), gradients(nullptr)
{
    this->type = sf::kLayerTypePooling;
    this->filterSize = 2;
    this->stride = 2;
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
    ulong width = this->inputWidth / this->filterSize;
    ulong height = this->inputHeight / this->filterSize;
    
    if (!(this->outputWidth == width && this->outputHeight == height))
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputWidth = width;
        this->outputHeight = height;
        this->outputDepth = this->inputDepth;
        
        ulong size = this->outputWidth * this->outputHeight;
        this->output = new double[size];
        this->selectedFilterIndexes = new unsigned char[size];
    }
    
    ulong outputSliceSize = this->outputWidth * this->outputHeight;
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
                
                auto index = outCol + (outRow * this->outputWidth) + (outputSliceSize * outLyr);
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
    if (this->gradients == nullptr)
    {
        this->gradients = new double[this->outputWidth * this->outputHeight * this->outputDepth];
    }
}