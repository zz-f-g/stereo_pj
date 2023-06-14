all : *.cpp
	g++ -g *.cpp -Wall `pkg-config --cflags --libs opencv`
