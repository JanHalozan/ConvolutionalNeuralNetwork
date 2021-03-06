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
    unsigned short stride;
    unsigned short kernelSide;
    unsigned char zeroPaddingSize;
    
    void resolveGradientCapacity();
    
public:
    ConvolutionLayer();
    ~ConvolutionLayer();
    
    void calculateOutput() override;
    void backprop(sf::Layer *previousLayer, sf::Layer *nextLayer) override;
    void recalculateWeights() override;
    
    void reserveNeurons(ulong count) override;
    double getGradientOfNeuron(ulong x, ulong y, ulong z) const override;
    
    void setStride(unsigned short stride);
    unsigned short getStride() const;
    
    void setKernelSideSize(unsigned short size);
    unsigned short getKernelSideSize() const;
    
    void setZeroPaddingSize(unsigned char size);
    unsigned char getZeroPaddingSize() const;
    
    void setOutputFeatureMapsCount(ulong count);
    ulong getOutputFeatureMapsCount();
};

#endif /* convolutionlayer_h */
