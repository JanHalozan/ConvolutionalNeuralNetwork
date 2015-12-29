//
//  convolutionlayer.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef convolutionlayer_h
#define convolutionlayer_h

#include <vector>

#include "neuron.h"
#include "layer.h"

namespace sf
{
    class ConvolutionLayer;
}

class sf::ConvolutionLayer : public sf::Layer
{
private:
    std::vector<sf::Neuron *> neurons;
    double convolve(double *input, double *kernel);
    
public:
    ConvolutionLayer();
    
    void calculateOutput() override;
    double *getOutput(unsigned long &width, unsigned long &height) override;
};

#endif /* convolutionlayer_h */
