//
//  loader.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 20/06/16.
//  Copyright © 2016 JanHalozan. All rights reserved.
//

#ifndef loader_h
#define loader_h

#include <fstream>
#include <string>
#include <vector>

class Loader {
    
private:
    std::vector<unsigned char> data;
    std::string cifarPath;
    
public:
    
    Loader(std::string cifarPath);
    ~Loader();
    
    void loadAllSamples();
    void getSampleAtIndex(unsigned int index, std::string &label, double *&sample);
    
};

#endif /* loader_h */
