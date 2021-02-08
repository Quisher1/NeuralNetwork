#include <random>
#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	
	NeuralNetwork net({2, 2, 1});

	net.setInput({1, 1});

	net.feedforward();


	std::cout << std::endl;
	for (auto k : net.getOutput()) {
		std::cout << k << " ";
	}std::cout << std::endl;

	return 0;
}