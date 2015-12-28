#include "layer.h"

sf::Layer::Layer()
{
}

sf::LayerType sf::Layer::getType()
{
    return this->type;
}

void sf::Layer::loadInput(double **input, unsigned short width, unsigned short height)
{
    this->inputWidth = width;
    this->inputHeight = height;
    this->input = input;
}
