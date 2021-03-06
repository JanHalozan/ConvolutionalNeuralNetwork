//
//  hiddenneuronlayer.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 02/01/16.
//  Copyright © 2016 JanHalozan. All rights reserved.
//

#ifndef hiddenneuronlayer_hpp
#define hiddenneuronlayer_hpp

#include "layer.h"

namespace sf
{
    class HiddenNeuronLayer;
}

class sf::HiddenNeuronLayer : public sf::Layer
{
private:
    
public:
    HiddenNeuronLayer();
    
    void calculateOutput() override;
    void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer) override;
    
    double getGradientOfNeuron(ulong x, ulong y, ulong z) const override;
};

#endif /* hiddenneuronlayer_hpp */
