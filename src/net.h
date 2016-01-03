//
//  net.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef net_h
#define net_h

#include "helpers.h"

#include <vector>
#include <limits>
#include <cmath>

#include "layer.h"
#include "outputneuronlayer.h"
#include "hiddenneuronlayer.h"
#include "poolinglayer.h"
#include "convolutionlayer.h"
#include "neuron.h"

namespace sf
{
    class Net;
}

class sf::Net
{
private:
    double breakErrorLimit;
    unsigned long breakEpochLimit;
    
    std::vector<sf::Layer *> layers;
    
    std::vector<double *> trainingSamples;
    std::vector<int> trainingSampleClasses;
    
    double *calculateNetOutput(double *sample);
    
public:
    unsigned long inputDataWidth;
    unsigned long inputDataHeight;
    
    Net(unsigned long inputDataWidth, unsigned long inputDataHeight);
    ~Net();
    
    void addLayer(sf::Layer *layer);
    
    void addTrainingSample(double *sample, int sampleClass);
    double *classifySample(double *sample);
    
    void train();
};

#endif /* net_h */
