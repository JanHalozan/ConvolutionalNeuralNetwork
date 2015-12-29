#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "layer.h"

namespace sf
{
    class PoolingLayer;
}

class sf::PoolingLayer : public sf::Layer
{
private:
    unsigned long poolingSize;
    
public:
    PoolingLayer();

    void calculateOutput() override;
    double *getOutput(unsigned long &width, unsigned long &height) override;
};

#endif // POOLINGLAYER_H
