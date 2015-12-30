//
//  layer.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef layer_h
#define layer_h

#include <vector>

#include "neuron.h"
#include "helpers.h"

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
}

class sf::Layer
{
protected:
    sf::LayerType type;
    
    std::vector<sf::Neuron> *neurons;
    
    unsigned long inputWidth;
    unsigned long inputHeight;
    
    unsigned long outputWidth;
    unsigned long outputHeight;
    
    double *input;
    double *output;
    
public:
    Layer();
    virtual ~Layer();
    
    sf::LayerType getType();
    
    void loadInput(double *input, unsigned long width, unsigned long height);
    virtual void calculateOutput() = 0;
    virtual double *getOutput(unsigned long &width, unsigned long &height);
    virtual void backprop() = 0;
    
    void reserveNeurons(unsigned long count);
};

#endif /* layer_h */
