#include "convolutionlayer.h"

sf::ConvolutionLayer::ConvolutionLayer() : Layer()
{
    
}

void sf::ConvolutionLayer::calculateOutput()
{
    
}

double *sf::ConvolutionLayer::getOutput(int &width, int &height)
{
    width = this->outputWidth;
    height = this->outputHeight;
    
    return this->output;
}