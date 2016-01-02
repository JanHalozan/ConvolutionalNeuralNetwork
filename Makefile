APPNAME = app

CC = gcc
CXX = g++
CXXFLAGS = -g -std=c++11 -Wall -pedantic
RM = rm -f
CPPFLAGS = -g
LDFLAGS = -g
LDLIBS =

SRCS = $(shell find . -maxdepth 2 -name "*.cpp")

build: clean
	mkdir app
	$(CXX) -x c++-header -o src/helpers.h.gch -c src/helpers.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o app/$(APPNAME) $(SRCS) $(LDLIBS)

clean:
	$(RM) -r app
