//
//  net.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include "net.h"

sf::Net::Net(ulong dataWidth, ulong dataHeight, ulong dataDepth) : breakErrorLimit(0.01), breakEpochLimit(100000), inputDataWidth(dataWidth), inputDataHeight(dataHeight), inputDataDepth(dataDepth)
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
    assert_log(this->layers.back()->getType() == kLayerTypeOutputNeuron, "Last layer must be of output type");
    
    //First let's find how many output neurons we need
    std::vector<int> uniqueClasses = this->trainingSampleClasses;
    std::sort(uniqueClasses.begin(), uniqueClasses.end());
    auto end = std::unique(uniqueClasses.begin(), uniqueClasses.end());
    const ulong uniqueClassesCount = std::distance(uniqueClasses.begin(), end);
    
    sf::Layer *layer = this->layers.back();
    layer->reserveNeurons(uniqueClassesCount);
    
    ulong epoch = 0;
    
    do
    {
        ulong sampleCounter = 0;
        double maxError = std::numeric_limits<double>::min();
        
        for (double *sample : this->trainingSamples)
        {
            //Get the net output. Every output of intermediate layers is now set.
            std::vector<double *> layeredOutput = this->calculateCompleteNetOutput(sample);
            double *output = layeredOutput.back();
            
            //We want the response to be 1.0 of the output neuron for the class we'were training now and 0.0 for all others.
            for (ulong i = 0; i < uniqueClassesCount; ++i)
            {
                double desiredResponse = sampleCounter == i ? 1.0 : 0.0;
                double error = fabs(desiredResponse - output[i]);
                
                if (error > maxError)
                    maxError = error;
            }
            
            auto layerOutputIt = layeredOutput.rend();
            for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it)
            {
                auto info = new sf::LayerBackpropInfo();
                info->samplesCount = this->trainingSamples.size();
                info->currentSampleNumber = sampleCounter;
                
                auto layer = *it;
                layer->loadInput(*layerOutputIt, this->inputDataWidth, this->inputDataHeight);
                layer->backprop(*(it + 1), *(it - 1), info);
                
                delete info;
                ++layerOutputIt;
            }
            
            for (auto layer : this->layers)
            {
                layer->recalculateWeights();
            }
            
            ++sampleCounter;
        }
        
        //TODO: Cleanup
        if (maxError < this->breakErrorLimit)
        {
            std::cout << "Minimum error rate reached in " << epoch << " epochs." << std::endl;
            break;
        }
        
        if (++epoch >= this->breakEpochLimit)
        {
            std::cout << "Epoch limit reached with " << maxError << " error rate." << std::endl;
            break;
        }
        
    } while (true);
}

double *sf::Net::classifySample(double *sample)
{
    return this->calculateNetOutput(sample);
}

std::vector<double *> sf::Net::calculateCompleteNetOutput(double *sample)
{
    std::vector<double *>layeredOutput;
    
    double *data = sample;
    ulong width = this->inputDataWidth;
    ulong height = this->inputDataHeight;
    ulong depth = this->inputDataDepth;
    
    for (auto layer : this->layers)
    {
        layer->loadInput(data, width, height, depth);
        layer->calculateOutput();
        data = layer->getOutput(width, height, depth);
        layeredOutput.push_back(data);
    }
    
    return layeredOutput;
}

double *sf::Net::calculateNetOutput(double *sample)
{
    double *data = sample;
    ulong width = this->inputDataWidth;
    ulong height = this->inputDataHeight;
    ulong depth = this->inputDataDepth;
    
    for (auto layer : this->layers)
    {
        layer->loadInput(data, width, height, depth);
        layer->calculateOutput();
        data = layer->getOutput(width, height, depth);
    }
    
    return data;
}