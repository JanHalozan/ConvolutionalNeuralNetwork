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
    ConvolutionLayer(unsigned long depth);
    
    unsigned short stride;
    unsigned short kernelSide;
    bool useZeroPadding;
    
    double convolve(double *input, double *kernel);
    
public:
    ConvolutionLayer();
    
    void calculateOutput() override;
    void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer, sf::LayerBackpropInfo *info) override;
    
    void setStride(unsigned short stride);
    unsigned short getStride() const;
    
    void setKernelSideSize(unsigned short size);
    unsigned short getKernelSideSize() const;
    
    void setUseZeroPadding(bool use);
    bool getUseZeroPadding() const;
};

#endif /* convolutionlayer_h */
