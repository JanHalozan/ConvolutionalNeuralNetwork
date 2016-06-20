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
#include <string>
#include <tuple>

#include "layers.h"
#include "neuron.h"

namespace sf
{
    class Net;
}

class sf::Net
{
private:
    double breakErrorLimit;
    ulong breakEpochLimit;
    
    std::vector<sf::Layer *> layers;
    
    std::vector<double *> trainingSamples;
    std::vector<std::string> trainingSampleClasses;
    std::vector<std::tuple<ulong, std::string>> sortedUniqueClasses;
    
    double *calculateNetOutput(double *sample);
    std::vector<double *> calculateCompleteNetOutput(double *sample);
    
public:
    ulong inputDataWidth;
    ulong inputDataHeight;
    ulong inputDataDepth;
    
    Net(ulong inputDataWidth, ulong inputDataHeight, ulong inputDataDepth = 1);
    ~Net();
    
    void addLayer(sf::Layer *layer);
    void addLayer(const sf::LayerDescriptor &descriptor);
    
    void addTrainingSample(double *sample, std::string sampleClass);
    std::vector<std::tuple<double, std::string>> classifySample(double *sample);
    
    void train();
    
    
};

#endif /* net_h */
