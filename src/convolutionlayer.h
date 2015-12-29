#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include <vector>

#include "neuron.h"
#include "layer.h"

namespace sf
{
    class ConvolutionLayer;
}

class sf::ConvolutionLayer : public sf::Layer
{
private:
    std::vector<sf::Neuron *> neurons;
    double convolve(double *input, double *kernel);
    
public:
    ConvolutionLayer();
    
    void calculateOutput() override;
    double *getOutput(int &width, int &height) override;
};

#endif // CONVOLUTIONLAYER_H
