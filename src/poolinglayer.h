#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "layer.h"

namespace sf
{
    class PoolingLayer;
}

class sf::PoolingLayer : public sf::Layer
{
public:
    PoolingLayer();

    void calculateOutput() override;
    double **getOutput(int &width, int &height) override;
};

#endif // POOLINGLAYER_H
