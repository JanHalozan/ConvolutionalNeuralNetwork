TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++11

SOURCES += src/main.cpp \
    src/neuron.cpp \
    src/layer.cpp \
    src/net.cpp \
    src/poolinglayer.cpp

HEADERS += \
    src/neuron.h \
    src/layer.h \
    src/net.h \
    src/poolinglayer.h
