//
//  poolinglayer.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef poolinglayer_h
#define poolinglayer_h

#include "layer.h"

namespace sf
{
    class PoolingLayer;
}

class sf::PoolingLayer : public sf::Layer
{
private:
    unsigned short stride;
    
    //Holds indexes that were selected during a forward pass so that we can backprop correctly
    unsigned char *selectedFilterIndexes;
    double *gradients;
    
public:
    PoolingLayer();
    ~PoolingLayer();

    void calculateOutput() override;
    void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer) override;
    
    double getGradientOfNeuron(ulong x, ulong y, ulong z) const override;
};

#endif /* poolinglayer_h */
