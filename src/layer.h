//
//  layer.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef layer_h
#define layer_h

#include "helpers.h"

#include <vector>

#include "neuron.h"

namespace sf
{
    class Layer;
    
    typedef enum
    {
        kLayerTypeConvolutional = 1,
        kLayerTypePooling = 2,
        kLayerTypeHiddenNeuron = 3,
        kLayerTypeOutputNeuron = 4
    } LayerType;
    
    struct LayerBackpropInfo
    {
        unsigned long samplesCount;
        unsigned long currentSampleNumber;
        
    };
}

class sf::Layer
{
protected:
    sf::LayerType type;
    
    std::vector<sf::Neuron> *neurons;
    
    unsigned long inputWidth;
    unsigned long inputHeight;
    unsigned long inputDepth;
    
    unsigned long outputWidth;
    unsigned long outputHeight;
    unsigned long outputDepth;
    
    double *input;
    double *output;
    
public:
    Layer();
    virtual ~Layer();
    
    sf::LayerType getType();
    const std::vector<sf::Neuron> getNeurons() const;
    
    void loadInput(double *input, unsigned long width, unsigned long height, unsigned long depth = 1);
    virtual void calculateOutput() = 0;
    virtual double *getOutput(unsigned long &width, unsigned long &height);
    
    virtual void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer, sf::LayerBackpropInfo *info) = 0;
    void recalculateWeights();
    
    void reserveNeurons(unsigned long count);
};

#endif /* layer_h */
