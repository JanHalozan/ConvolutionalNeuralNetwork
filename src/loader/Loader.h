//
//  Loader.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 20/06/16.
//  Copyright © 2016 JanHalozan. All rights reserved.
//

#ifndef Loader_h
#define Loader_h

#include <fstream>
#include <string>
#include <vector>

class Loader {
    
private:
    std::ifstream file;
    std::vector<unsigned char> data;
    
public:
    
    Loader(std::string cifarPath);
    ~Loader();
    
    void loadAllSamples();
    void getSampleAtIndex(unsigned int index, double *sample);
    
};

#endif /* Loader_h */
