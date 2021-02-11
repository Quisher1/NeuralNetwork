#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <random>

class NeuralNetwork {
public:
	NeuralNetwork();
	explicit NeuralNetwork(std::vector<int> &neuronLayers);

	void init(std::vector<int> &neuronLayers);

	void setInput(std::vector<float> input);
	void setOutput(std::vector<float> output);
	void setLearningRate(float lRate);

	void randomizeWeights();

	void feedforward();

	float train();


	void loadWeightsFromTxtFile(const std::string &filename);
	void saveWeightsToTxtFile(const std::string &filename);

	std::vector<float> getOutput() const;
	float getError();


	void printNeurons() const;
	void printInput() const;
	void printOut() const;
	void printWeights() const;

private:
	int N;
	float learningRate = 0.1f;

	std::vector<int> neuronLayers;

	std::vector<float> input;
	std::vector<float> output;

	std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<float>> neurons;

	std::vector<std::vector<float>> deltaNeurons;


	float sigmoid(float x) const;
	float sigmoidDerivative(float x) const;
};