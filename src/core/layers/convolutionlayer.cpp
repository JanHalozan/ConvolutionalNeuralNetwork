//
//  convolutionlayer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "convolutionlayer.h"

#include "poolinglayer.h"

sf::ConvolutionLayer::ConvolutionLayer() : Layer(), stride(1), kernelSide(3), zeroPaddingSize(0)
{
    this->type = kLayerTypeConvolutional;
}

sf::ConvolutionLayer::~ConvolutionLayer()
{
}

void sf::ConvolutionLayer::calculateOutput()
{
    assert_log(ceil((this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride) == (this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride, "Invalid hyper parameters (width)");
    assert_log(ceil((this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride) == (this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride, "Invalid hyper parameters (height)");
    
    const ulong oWidth = (this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride + 1;
    const ulong oHeight = (this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride + 1;
    
    if (!(this->outputWidth == oWidth && this->outputHeight == oHeight))
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputWidth = oWidth;
        this->outputHeight = oHeight;
        this->output = new double[this->outputWidth * this->outputHeight * this->outputDepth];
    }
    
    const ulong sliceSize = this->inputWidth * this->inputHeight;
    const ulong outputSliceSize = this->outputWidth * this->outputHeight;
    ulong outRow = 0;
    ulong outCol = 0;
    ulong outLyr = 0;
    
    for (auto &neuron : *this->neurons)
    {
        for (llong row = -this->zeroPaddingSize; row < (signed)(this->inputHeight - this->kernelSide + this->zeroPaddingSize + 1); row += this->stride)
        {
            for (llong col = -this->zeroPaddingSize; col < (signed)(this->inputWidth - this->kernelSide + this->zeroPaddingSize + 1); col += this->stride)
            {
                std::vector<double> input(this->kernelSide * this->kernelSide * this->inputDepth);
                ulong i = 0;
                
                for (ulong z = 0; z < this->inputDepth; ++z)
                {
                    for (ulong y = 0; y < this->kernelSide; ++y)
                    {
                        for (ulong x  = 0; x < this->kernelSide; ++x)
                        {
                            if (row < 0 || row >= (signed)this->inputHeight || col < 0 || col >= (signed)this->inputWidth)
                            {
                                input[i++] = 0.0;
                                continue;
                            }
                            
                            const double val = this->input[(col + x) + ((row + y) * this->inputWidth) + (z * sliceSize)];
                            input[i++] = val;
                        }
                    }
                }
                
                neuron.loadInput(input);
                neuron.calculateOutput();
                
                this->output[outCol + (outRow * this->outputWidth) + (outLyr * outputSliceSize)] = neuron.getOutput();
                
                ++outCol;
                outRow += outCol / this->outputWidth; //Adds 1 only when actually entire row has been filled (C integer division)
                outLyr += (outRow / this->outputHeight) * (outCol / this->outputWidth); //Same as above (only 1 * 1 = 1)
                outCol %= this->outputWidth;
                outRow %= this->outputHeight;
            }
        }
    }
}

void sf::ConvolutionLayer::backprop(sf::Layer *, sf::Layer *nextLayer)
{
    for (ulong lyr = 0; lyr < this->outputDepth; ++lyr)
    {
        const auto &neuron = this->neurons->at(lyr);
        for (ulong row = 0; row < this->outputHeight; ++row)
        {
            for (ulong col = 0; col < this->outputWidth; ++col)
            {
                double gradient = 0;
                
                
            }
        }
    }
    
    
////    const ulong gradientsSize = this->outputWidth * this->outputHeight * this->outputDepth;
////    if (this->gradients == nullptr)
////        this->gradients = new double[gradientsSize];
////    
////    memset(this->gradients, 0, gradientsSize * sizeof(double));
//    
//    auto kernelSize = this->kernelSide * this->kernelSide;
//    auto outputSliceSize = this->outputWidth * this->outputHeight;
//    
//    //TODO: Incorporate stride and other things?
//    for (ulong lyr = 0; lyr < this->outputDepth; ++lyr)
//    {
//        for (ulong row = 0; row < this->outputHeight; ++row)
//        {
//            for (ulong col = 0; col < this->outputWidth; ++col)
//            {
//                const ulong index = col + (row * this->outputWidth) + (lyr * outputSliceSize);
//                auto &neuron = this->neurons->at(lyr);
//                
//                //TODO: Here you must get gradients & weights in the next layer which used this neuron (actual in the 2D grid - not in code; code uses 1 neuron for entire slice) to calculate their output
//                const ulong matrixSize = 9; //Next layer size of matrix for covolution
//                const double *weights = nullptr; //TODO
//                const double *gradients = nullptr; //TODO
//                
//                double val = 0.0;
//                
//                for (ulong i = 0; i < matrixSize; ++i)
//                    val += gradients[i] * weights[matrixSize - 1 - i];
//                
//                const double outGradient = neuron.getOutput() * (1.0 - neuron.getOutput()) * val;
//                const ulong gradIndex = col + (row * this->outputWidth);
//                neuron.setGradient(outGradient, gradIndex);
//                
////                const ulong index = col + (row * this->outputWidth) + (lyr * outputSliceSize);
////                const double gradient = nextLayer->getGradientOfNeuron(index); //REMOVED
////                double outGradient = 0.0;
////                auto &n = this->neurons->at(lyr);
////                
////                //Optimization from polling layer - polling rutes only the max gradient//NOT SURE YET, CHAIN RULE SHIT - check
////                if (gradient != 0.0)
////                {
////                    auto &&weights = n.getWeights();
////                    double val = 0.0;
////                    
////                    for (unsigned short i = 0; i < kernelSize; ++i)
////                        val += weights[kernelSize - i - 1] * gradient;
////                    
////                    outGradient = n.getOutput() * (1.0 - n.getOutput()) * val;
////                }
////                
////                n.setGradient(outGradient);
//////                this->gradients[index] = outGradient;
//            }
//        }
//    }
}

void sf::ConvolutionLayer::recalculateWeights()
{
    for (ulong lyr = 0; lyr < this->outputDepth; ++lyr)
    {
        for (ulong row = 0; row < this->outputHeight; ++row)
        {
            for (ulong col = 0; col < this->outputWidth; ++col)
            {
                auto &neuron = this->neurons->at(lyr);
                
            }
        }
    }
}

void sf::ConvolutionLayer::reserveNeurons(ulong count)
{
    Layer::reserveNeurons(count);
    
    //+1 for the bias weight
    const ulong inputCount = this->kernelSide * this->kernelSide * this->inputDepth + 1;
    
    for (sf::Neuron &n : *this->neurons)
        n.randomizeWeights(inputCount);
    
    this->resolveGradientCapacity();
}

double sf::ConvolutionLayer::getGradientOfNeuron(ulong x, ulong y, ulong z) const
{
    const auto &neuron = this->neurons->at(z);
    const ulong pos = x + this->outputWidth * y;
    
    return neuron.getGradient(pos);
}

void sf::ConvolutionLayer::resolveGradientCapacity()
{
    const ulong oWidth = (this->inputWidth - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride + 1;
    const ulong oHeight = (this->inputHeight - this->kernelSide + 2 * this->zeroPaddingSize) / this->stride + 1;
    const ulong sliceSize = this->outputWidth * this->outputHeight;
    
    if (!(this->outputWidth == oWidth && this->outputHeight == oHeight))
    {
        if (this->output != nullptr)
            delete[] this->output;
        
        this->outputWidth = oWidth;
        this->outputHeight = oHeight;
        
        const ulong outputSize = sliceSize * this->outputDepth;
        this->output = new double[outputSize];
    }
    
    for (auto &n : *this->neurons)
        n.reserveGradientItems(sliceSize);
}

void sf::ConvolutionLayer::setStride(unsigned short stride)
{
    this->stride = stride;
    this->resolveGradientCapacity();
}

unsigned short sf::ConvolutionLayer::getStride() const
{
    return this->stride;
}

void sf::ConvolutionLayer::setKernelSideSize(unsigned short size)
{
    this->kernelSide = size;
    
    //+1 for the bias weight
    const ulong inputCount = this->kernelSide * this->kernelSide + 1;
    
    for (auto &n : *this->neurons)
        n.randomizeWeights(inputCount);
    
    this->resolveGradientCapacity();
}

unsigned short sf::ConvolutionLayer::getKernelSideSize() const
{
    return this->kernelSide;
}

void sf::ConvolutionLayer::setZeroPaddingSize(unsigned char size)
{
    this->zeroPaddingSize = size;
    this->resolveGradientCapacity();
}

unsigned char sf::ConvolutionLayer::getZeroPaddingSize() const
{
    return this->zeroPaddingSize;
}

void sf::ConvolutionLayer::setOutputFeatureMapsCount(ulong count)
{
    this->outputDepth = count;
    this->reserveNeurons(count);
    this->resolveGradientCapacity();
}

ulong sf::ConvolutionLayer::getOutputFeatureMapsCount()
{
    return this->outputDepth;
}
