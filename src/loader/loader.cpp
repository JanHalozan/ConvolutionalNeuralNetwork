//
//  loader.cpp
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 20/06/16.
//  Copyright © 2016 JanHalozan. All rights reserved.
//

#include "loader.h"

//153650000 is the size of all 5 files for test images
Loader::Loader(std::string filePath) : data(153650000), cifarPath(filePath)
{
}

Loader::~Loader()
{
}

void Loader::loadAllSamples()
{
    char c;
    unsigned int i = 0;
    std::string files[] = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"};
    
    for (auto &fileName : files)
    {
        std::ifstream file(this->cifarPath + fileName, std::ios::binary | std::ios::in);
        
        while (file.get(c))
        {
            this->data[i] = (unsigned char)c;
            ++i;
        }
        
        file.close();
    }
}

void Loader::getSampleAtIndex(unsigned int index, std::string &label, double *&sample)
{
    const size_t sampleSize = 3072;
    sample = new double[sampleSize];
    auto startOffset = index * 3073;
    
    label = "test";
    
    startOffset++;
    memcpy(sample, this->data.data() + startOffset, sampleSize);
}