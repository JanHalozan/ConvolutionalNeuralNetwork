//
//  convolutionlayer.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef convolutionlayer_h
#define convolutionlayer_h

#include "layer.h"

namespace sf
{
    class ConvolutionLayer;
}

class sf::ConvolutionLayer : public sf::Layer
{
private:
    double convolve(double *input, double *kernel);
    
public:
    ConvolutionLayer();
    
    void calculateOutput() override;
    void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer, sf::LayerBackpropInfo *info) override;
};

#endif /* convolutionlayer_h */
