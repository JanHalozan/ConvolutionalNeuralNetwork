//
//  net.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "net.h"

sf::Net::Net(unsigned long dataWidth, unsigned long dataHeight) : breakErrorLimit(0.001), breakEpochLimit(ULONG_MAX), inputDataWidth(dataWidth), inputDataHeight(dataHeight)
{
}

sf::Net::~Net()
{
    for (sf::Layer *layer : this->layers)
        delete layer;
}

void sf::Net::addLayer(sf::Layer *layer)
{
    this->layers.push_back(layer);
}

void sf::Net::addTrainingSample(double *sample, int sampleClass)
{
    this->trainingSamples.push_back(sample);
    this->trainingSampleClasses.push_back(sampleClass);
}

void sf::Net::train()
{
    assert_log(this->layers.back()->getType() != kLayerTypeOutputNeuron, "Last layer must be of output type");
    
    //First let's find how many output neurons we need
    std::vector<int> uniqueClasses = this->trainingSampleClasses;
    std::sort(uniqueClasses.begin(), uniqueClasses.end());
    auto end = std::unique(uniqueClasses.begin(), uniqueClasses.end());
    const unsigned long uniqueClassesCount = std::distance(uniqueClasses.begin(), end);
    
    sf::Layer *layer = this->layers.back();
    layer->reserveNeurons(uniqueClassesCount);
    
    unsigned long epoch = 0;
    
    do
    {
        std::vector<double> errors(uniqueClassesCount, -1.0);
        unsigned long sampleCounter = 0;
        double maxError = std::numeric_limits<double>::min();
        
        for (double *sample : this->trainingSamples)
        {
            //Get the net output. Every output of intermediate layers is now set.
            double *output = this->calculateNetOutput(sample);
            
            //We want the response to be 1.0 of the output neuron for the class we'were training now and 0.0 for all others.
            for (unsigned long i = 0; i < uniqueClassesCount; ++i)
            {
                double desiredResponse = sampleCounter == i ? 1.0 : 0.0;
                double error = fabs(desiredResponse - output[i]);
                
                if (error > maxError)
                    maxError = error;
            }
            
            for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it)
            {
                auto info = new sf::LayerBackpropInfo();
                info->samplesCount = this->trainingSamples.size();
                info->currentSampleNumber = sampleCounter;
                
                auto layer = *it;
                layer->loadInput(sample, this->inputDataWidth, this->inputDataHeight);
                layer->backprop(*(it - 1), *(it + 1), info);
                
                delete info;
            }
            
            ++sampleCounter;
        }
        
        //TODO: Cleanup
        if (maxError < this->breakErrorLimit)
            break;
        if (++epoch >= this->breakEpochLimit)
            break;
        
    } while (true);
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