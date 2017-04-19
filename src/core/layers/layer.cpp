//
//  layer.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "layer.h"

sf::Layer::Layer() : inputWidth(0), inputHeight(0), inputDepth(0)
{
    this->output = nullptr;
    this->input = nullptr;
    this->neurons = new std::vector<sf::Neuron>();
}

sf::Layer::~Layer()
{
    delete this->neurons;
}

sf::LayerType sf::Layer::getType() const
{
    return this->type;
}

void sf::Layer::reserveNeurons(ulong count)
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
            {
                sf::Neuron n;
                n.activationType = this->type == kLayerTypeConvolutional ? kNeuronActivationFunctionTypeConvolution : kNeuronActivationFunctionTypeSig;
                this->neurons->push_back(n);
            }
        }
    }
}

const std::vector<sf::Neuron> sf::Layer::getNeurons() const
{
    return *this->neurons;
}

void sf::Layer::loadInput(double *input, ulong width, ulong height, ulong depth)
{
    this->inputWidth = width;
    this->inputHeight = height;
    this->inputDepth = depth;
    this->input = input;
}

double *sf::Layer::getOutput(ulong &width, ulong &height, ulong &depth) const
{
    width = this->outputWidth;
    height = this->outputHeight;
    depth = this->outputDepth;
    
    return this->output;
}

void sf::Layer::recalculateWeights()
{
    for (auto &n : *this->neurons)
        n.recalculateWeights();
}

void sf::Layer::setInputWidth(ulong w)
{
    this->inputWidth = w;
}

ulong sf::Layer::getInputWidth() const
{
    return this->inputWidth;
}

void sf::Layer::setInputHeight(ulong h)
{
    this->inputHeight = h;
}

ulong sf::Layer::getInputHeight() const
{
    return this->inputHeight;
}

void sf::Layer::setInputDepth(ulong d)
{
    this->inputDepth = d;
}

ulong sf::Layer::getInputDepth() const
{
    return this->inputDepth;
}
