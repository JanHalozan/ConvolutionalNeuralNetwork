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