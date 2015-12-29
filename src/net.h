#ifndef NET_H
#define NET_H

#include <vector>

#include "layer.h"

namespace sf
{
    class Net;
}

class sf::Net
{
private:
    std::vector<sf::Layer *> layers;
    std::vector<double *> trainingSamples;
    
    double *calculateNetOutput(double *input);
    
public:
    static unsigned long inputDataWidth;
    static unsigned long inputDataHeight;
    
    Net();
    
    void addLayer(sf::Layer *layer);
    
    void addTrainingSample(double *data);
    double *classifySample(double *data);
    
    void train();
};

#endif // NET_H
