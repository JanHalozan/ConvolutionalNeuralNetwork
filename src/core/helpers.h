//
//  helpers.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef helpers_h
#define helpers_h

typedef unsigned long ulong;
typedef long long llong;

#define DEBUG

#ifdef DEBUG

#define GCC diagnostic ignored "-Wkeyword-macro"

#include <iostream>
#include <cassert>

#define assert_log(x, y) {if (!(x)) {std::cout << y << std::endl; assert(false);}}

#define private public
#define protected public

#else

#define assert_log(x, y)

#endif

#endif /* helpers_h */
