//
//  net.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "net.h"

unsigned long sf::Net::inputDataWidth = 5;
unsigned long sf::Net::inputDataHeight = 1;

sf::Net::Net()
{
}

void sf::Net::addLayer(sf::Layer *layer)
{
    this->layers.push_back(layer);
}

void sf::Net::addTrainingSample(double *sample)
{
    this->trainingSamples.push_back(sample);
}

void sf::Net::train()
{
    
}

double *sf::Net::classifySample(double *sample)
{
    return this->calculateNetOutput(sample);
}

double *sf::Net::calculateNetOutput(double *sample)
{
    double *data = sample;
    unsigned long width = this->inputDataWidth;
    unsigned long height = this->inputDataHeight;
    
    for (sf::Layer *layer : this->layers)
    {
        layer->loadInput(data, width, height);
        layer->calculateOutput();
        data = layer->getOutput(width, height);
    }
    
    return data;
}