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
    unsigned long poolingSize;
    
public:
    PoolingLayer();

    void calculateOutput() override;
    double *getOutput(unsigned long &width, unsigned long &height) override;
};

#endif /* poolinglayer_h */
