#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include "layer.h"

namespace sf
{
    class ConvolutionLayer;
}

class sf::ConvolutionLayer : public sf::Layer
{
private:
    
public:
    ConvolutionLayer();
    
    void calculateOutput() override;
    double *getOutput(int &width, int &height) override;
};

#endif // CONVOLUTIONLAYER_H
