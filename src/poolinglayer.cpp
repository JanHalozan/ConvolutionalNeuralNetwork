#include "poolinglayer.h"

sf::PoolingLayer::PoolingLayer() : Layer()
{
    this->type = sf::kLayerTypePooling;
    this->poolingSize = 2;
}

void sf::PoolingLayer::calculateOutput()
{
    if (this->input == nullptr)
        return;
    
    if (this->output != nullptr)
        delete[] this->output;
    
    //Both must be a power of 2
    this->outputWidth = this->inputWidth / this->poolingSize;
    this->outputHeight = this->inputHeight / this->poolingSize;
    this->output = new double[this->outputWidth * this->outputHeight];
    
    int outRow = 0;
    int outCol = 0;
    
    //Goes first columns then rows
    for (int row = 0; row < this->inputHeight; row += this->poolingSize)
    {
        for (int col = 0; col < this->inputWidth; col += this->poolingSize)
        {
            double max = maxQuartet(this->input[row * this->inputWidth + col],
                                    this->input[row * this->inputWidth + col + 1],
                                    this->input[(row + 1) * this->inputWidth + col],
                                    this->input[(row + 1) * this->inputWidth + col + 1]);
            this->output[outRow * this->outputWidth + outCol] = max;
            
            ++outCol;
            outRow += outCol / this->outputWidth;
            outCol %= this->outputWidth;
        }
    }
}

double *sf::PoolingLayer::getOutput(int &width, int &height)
{
    width = this->outputWidth;
    height = this->outputHeight;
    
    return this->output;
}
