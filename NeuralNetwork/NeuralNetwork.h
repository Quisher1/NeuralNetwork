#pragma once
#include <vector>
#include <iostream>
#include <random>

class NeuralNetwork {
public:
	explicit NeuralNetwork(std::vector<int> neuronLayers);

	void setInput(std::vector<float> input);
	void setOutput(std::vector<float> output);

	void randomizeWeights();

	void feedforward();

	float train();


	std::vector<float> getOutput();

private:
	int N;
	std::vector<int> neuronLayers;

	std::vector<float> input;
	std::vector<float> output;

	std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<float>> neurons;

	float sigmoidFunction(float x);
};