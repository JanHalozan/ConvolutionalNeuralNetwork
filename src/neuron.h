#ifndef NEURON_H
#define NEURON_H

#define NEURON_PARAMETER_SLOPE 1.0

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
    
public:
    Neuron();
    
    void randomizeWeights();
    void loadInput(std::vector<double> input);
    void calculateOutput();
    double getOutput();
};

#endif // NEURON_H
