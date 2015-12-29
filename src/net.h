//
//  net.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef net_h
#define net_h

#include <vector>

#include "layer.h"

namespace sf
{
    class Net;
}

class sf::Net
{
private:
    std::vector<sf::Layer *> layers;
    std::vector<double *> trainingSamples;
    
    double *calculateNetOutput(double *sample);
    
public:
    static unsigned long inputDataWidth;
    static unsigned long inputDataHeight;
    
    Net();
    
    void addLayer(sf::Layer *layer);
    
    void addTrainingSample(double *sample);
    double *classifySample(double *sample);
    
    void train();
};

#endif /* net_h */
