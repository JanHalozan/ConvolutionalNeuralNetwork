#ifndef LAYER_H
#define LAYER_H

#include "helpers.h"

namespace sf
{
    class Layer;
    
    typedef enum
    {
        kLayerTypeConvolutional = 1,
        kLayerTypePooling = 2,
        kLayerTypeNeural = 3
    } LayerType;
}

class sf::Layer
{
protected:
    sf::LayerType type;
    
    unsigned long inputWidth;
    unsigned long inputHeight;
    
    unsigned long outputWidth;
    unsigned long outputHeight;
    
    double *input;
    double *output;
    
public:
    Layer();
    
    sf::LayerType getType();
    
    void loadInput(double *input, unsigned long width, unsigned long height);
    virtual void calculateOutput() = 0;
    virtual double *getOutput(unsigned long &width, unsigned long &height) = 0;
};

#endif // LAYER_H
