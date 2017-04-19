//
//  outputneuronlayer.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef outputneuronlayer_h
#define outputneuronlayer_h

#include "layer.h"

namespace sf
{
    class OutputNeuronLayer;
}

class sf::OutputNeuronLayer : public sf::Layer
{
private:
    ulong backpropTargetNeuron;
    
public:
    OutputNeuronLayer();
    
    void setBackpropTargetNeuron(ulong index);
    
    void calculateOutput() override;
    void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer) override;
    
    double getGradientOfNeuron(ulong x, ulong y = 0, ulong z = 0) const override;
};

#endif /* outputneuronlayer_h */
