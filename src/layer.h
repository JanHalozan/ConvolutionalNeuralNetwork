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
        ulong samplesCount;
        ulong currentSampleNumber;

    };
}

class sf::Layer
{
protected:
    sf::LayerType type;

    std::vector<sf::Neuron> *neurons;

    ulong inputWidth;
    ulong inputHeight;
    ulong inputDepth;

    ulong outputWidth;
    ulong outputHeight;
    ulong outputDepth;

    double *input;
    double *output;

public:
    Layer();
    virtual ~Layer();

    sf::LayerType getType();
    const std::vector<sf::Neuron> getNeurons() const;

    void loadInput(double *input, ulong width, ulong height, ulong depth = 1);
    virtual void calculateOutput() = 0;
    virtual double *getOutput(ulong &width, ulong &height, ulong &depth);

    virtual void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer, sf::LayerBackpropInfo *info) = 0;
    void recalculateWeights();

    virtual void reserveNeurons(ulong count);
};

#endif /* layer_h */
