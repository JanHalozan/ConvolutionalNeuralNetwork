#ifndef NEURON_H
#define NEURON_H

#include <vector>

namespace sf
{
    class Neuron;
}

class sf::Neuron
{
private:
    std::vector<double> weights;

    double threshold;
    


public:
    Neuron();
};

#endif // NEURON_H
