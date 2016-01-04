//
//  layer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "layer.h"

sf::Layer::Layer() : inputWidth(0), inputHeight(0)
{
    this->output = nullptr;
    this->input = nullptr;
    this->neurons = new std::vector<sf::Neuron>();
}

sf::Layer::~Layer()
{
    delete this->neurons;
}

sf::LayerType sf::Layer::getType()
{
    return this->type;
}

const std::vector<sf::Neuron> sf::Layer::getNeurons() const
{
    return *this->neurons;
}

void sf::Layer::loadInput(double *input, unsigned long width, unsigned long height)
{
    this->inputWidth = width;
    this->inputHeight = height;
    this->input = input;
}

double *sf::Layer::getOutput(unsigned long &width, unsigned long &height)
{
    width = this->outputWidth;
    height = this->outputHeight;
    
    return this->output;
}

void sf::Layer::recalculateWeights()
{
    for (auto &n : *this->neurons)
        n.recalculateWeights();
}

void sf::Layer::reserveNeurons(unsigned long count)
{
    if (this->neurons->size() != count)
    {
        const long diff = labs((long)(this->neurons->size() - count));
        if (this->neurons->size() > count)
        {
            this->neurons->erase(this->neurons->end() - diff, this->neurons->end());
        }
        else
        {
            for (long i = 0; i < diff; ++i)
                this->neurons->push_back(sf::Neuron());
        }
    }
}