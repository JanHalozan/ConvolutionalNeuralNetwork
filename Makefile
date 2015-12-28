APPNAME = app

CC = gcc
CXX = g++
RM = rm -f
CPPFLAGS = -g
LDFLAGS = -g
LDLIBS =

SRCS = $(shell find . -maxdepth 2 -name "*.cpp")

build:
	mkdir app
	$(CXX) $(LDFLAGS) -o app/$(APPNAME) $(SRCS) $(LDLIBS)

clean:
	$(RM) -r app
