//
//  helpers.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef helpers_h
#define helpers_h

template <typename T> T maxQuartet(T a, T b, T c, T d)
{
    T val = a;
    if (b > val) val = b;
    if (c > val) val = c;
    if (d > val) val = d;
    
    return val;
}

#endif /* helpers_h */
