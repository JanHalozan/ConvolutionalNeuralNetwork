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

public:
    Net();
};

#endif // NET_H
