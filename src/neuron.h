#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "helpers.h"

namespace sf
{
    class Neuron;
}

class sf::Neuron
{
private:
    std::vector<double> weights;
    std::vector<double> inputs;

    double output;
    double gradient;
    
public:
    static double learningRate;
    
    Neuron();
    
    void randomizeWeights();
    void loadInput(std::vector<double> input);
    void calculateOutput();
    double getOutput();
    
    void backpop(std::vector<double> values);
    double getGradient();
};

#endif // NEURON_H
