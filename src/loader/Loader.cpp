//
//  Loader.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 20/06/16.
//  Copyright © 2016 JanHalozan. All rights reserved.
//

#include "Loader.h"

//153650000 is the size of all 5 files for test images
Loader::Loader(std::string filePath) : file(filePath, std::ios::binary | std::ios::in), data(153650000)
{
}

void Loader::loadAllSamples()
{
    char c;
    unsigned int i = 0;
    while (this->file.get(c))
    {
        this->data[i] = (unsigned char)c;
        ++i;
    }
}