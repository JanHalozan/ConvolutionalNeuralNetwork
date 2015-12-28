#ifndef LAYER_H
#define LAYER_H

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

    unsigned short inputWidth;
    unsigned short inputHeight;

    unsigned short outputWidth;
    unsigned short outputHeight;

    double **input;
    double **output;

public:
    Layer();

    sf::LayerType getType();

    void loadInput(double **input, unsigned short width, unsigned short height);
    virtual void calculateOutput() = 0;
    virtual double **getOutput(int &width, int &height) = 0;
};

#endif // LAYER_H
