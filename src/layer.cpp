#include "layer.h"

sf::Layer::Layer()
{
    this->output = nullptr;
    this->input = nullptr;
}

sf::LayerType sf::Layer::getType()
{
    return this->type;
}

void sf::Layer::loadInput(double *input, unsigned long width, unsigned long height)
{
    this->inputWidth = width;
    this->inputHeight = height;
    this->input = input;
}
