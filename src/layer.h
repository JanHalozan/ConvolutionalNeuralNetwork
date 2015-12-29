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
    
    unsigned long inputWidth;
    unsigned long inputHeight;
    
    unsigned long outputWidth;
    unsigned long outputHeight;
    
    double *input;
    double *output;
    
public:
    Layer();
    
    sf::LayerType getType();
    
    void loadInput(double *input, unsigned long width, unsigned long height);
    virtual void calculateOutput() = 0;
    virtual double *getOutput(unsigned long &width, unsigned long &height) = 0;
};

#endif /* layer_h */
