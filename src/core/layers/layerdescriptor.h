//
//  layerdescriptor.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 17/01/16.
//  Copyright © 2016 JanHalozan. All rights reserved.
//

#ifndef layerdescriptor_h
#define layerdescriptor_h

#include "helpers.h"

#include "layer.h"
#include "outputneuronlayer.h"
#include "hiddenneuronlayer.h"
#include "poolinglayer.h"
#include "convolutionlayer.h"

namespace sf
{
    struct LayerDescriptor;
    inline Layer *makeLayer(const LayerDescriptor &descriptor);
}

struct sf::LayerDescriptor
{
    LayerType type;
    
    ulong inputWidth;
    ulong inputHeight;
    ulong inputDepth;
    
    ulong outputWidth;
    ulong outputHeight;
    ulong outputDepth;
    
    ulong neuronCount;
    
    unsigned short stride;
    unsigned short kernelSide;
    
};

sf::Layer *sf::makeLayer(const sf::LayerDescriptor &d)
{
    Layer *ret = nullptr;
    
    switch (d.type)
    {
        case kLayerTypeConvolutional:
        {
            ConvolutionLayer *layer = new ConvolutionLayer();

            layer->setOutputFeatureMapsCount(d.outputDepth);
            
            ret = layer;
        }
            break;
        case kLayerTypePooling:
        {
            PoolingLayer *layer = new PoolingLayer();
            
            ret = layer;
        }
            break;
        case kLayerTypeHiddenNeuron:
        {
            HiddenNeuronLayer *layer = new HiddenNeuronLayer();
            layer->reserveNeurons(d.neuronCount);
            
            ret = layer;
        }
            break;
        case kLayerTypeOutputNeuron:
        {
            OutputNeuronLayer *layer = new OutputNeuronLayer();
            
            ret = layer;
        }
            break;
    }
    
    ret->setInputWidth(d.inputWidth);
    ret->setInputHeight(d.inputHeight);
    ret->setInputDepth(d.inputDepth);
    
    return ret;
}

#endif /* layerdescriptor_h */
