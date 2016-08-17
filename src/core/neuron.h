//
//  neuron.h
//  ConvolutionalNeuralNetwork
//
//  Created by Jan Haložan on 29/12/15.
//  Copyright © 2015 JanHalozan. All rights reserved.
//

#ifndef neuron_h
#define neuron_h

#include "helpers.h"

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

namespace sf
{
    class Neuron;
    
    typedef enum {
        kNeuronActivationFunctionTypeSig = 1,
        kNeuronActivationFunctionTypeConvolution = 2
    } NeuronActivationFunctionType;
}

class sf::Neuron
{
private:
    sf::NeuronActivationFunctionType activationType;
    
    std::vector<double> weights;
    std::vector<double> inputs;
    std::vector<double> gradients;

    double output;
    
public:
    static double learningRate;
    
    Neuron();
    
    void randomizeWeights(const long count);
    
    double getWeight(const ulong index) const;
    const std::vector<double> getWeights() const;
    
    void loadInput(std::vector<double> input);
    void calculateOutput();
    double getOutput() const;
    
    void reserveGradientItems(const ulong count);
    void setGradient(const double g, const ulong index = 0);
    double getGradient(const ulong index = 0) const;
    
    void recalculateWeights(const ulong gradientIndex = 0);
    
    void setActivationFunctionType(const sf::NeuronActivationFunctionType t);
    sf::NeuronActivationFunctionType getActivationFunctionType() const;
};

#endif /* neuron_h */
