//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#include <cstdlib>
#include <cassert>
#include <iostream>

#include "poolinglayer.h"

int main(int argc, char const *argv[])
{
    sf::PoolingLayer *layer = new sf::PoolingLayer();
    double data[] = {
        4, 5, 4, 5,
        6, 7, 8, 7,
        2, 3, 6, 1,
        9, 6, 5, 4
    };
    
    layer->loadInput(data, 4, 4);
    layer->calculateOutput();
    
    unsigned long w, h;
    double *res = layer->getOutput(w, h);
    assert(w == 2);
    assert(h == 2);
    assert(res[0] == 7);
    assert(res[1] == 8);
    assert(res[2] == 9);
    assert(res[3] == 6);

    std::cout << "All good" << std::endl;
    
    return 0;
}
