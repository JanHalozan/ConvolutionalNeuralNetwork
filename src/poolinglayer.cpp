#include "poolinglayer.h"

sf::PoolingLayer::PoolingLayer()
{
    this->type = sf::kLayerTypePooling;
}

void sf::PoolingLayer::calculateOutput()
{

}

double **sf::PoolingLayer::getOutput(int &width, int &height)
{
    width = this->outputWidth;
    height = this->outputHeight;

    return this->output;
}
