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

void sf::Net::addTrainingSample(double *data)
{
    this->trainingSamples.push_back(data);
}

void sf::Net::train()
{
    
}

double *sf::Net::classifySample(double *data)
{
    double *input = data;
    unsigned long width = this->inputDataWidth;
    unsigned long height = this->inputDataHeight;
    
    for (sf::Layer *layer : this->layers)
    {
        layer->loadInput(input, width, height);
        layer->calculateOutput();
        input = layer->getOutput(width, height);
    }
    
    return NULL;
}