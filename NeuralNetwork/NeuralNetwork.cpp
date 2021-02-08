#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int> neuronLayers) {
	N = neuronLayers.size();
	this->neuronLayers = neuronLayers;

	neurons.resize(N);
	weights.resize(N - 1);

	for (int i = 0; i < N; ++i)
		neurons[i].resize(neuronLayers[i]);

	for (int i = 0; i < N - 1; ++i) {
		weights[i].resize(neuronLayers[i]);
		for (int j = 0; j < neuronLayers[i]; ++j) {
			weights[i][j].resize(neuronLayers[i + 1]);
		}
	}

	randomizeWeights();

	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < neuronLayers[i]; ++j) {
			for (int k = 0; k < neuronLayers[i + 1]; ++k) {
				std::cout << weights[i][j][k] << " ";
			}std::cout << std::endl;
		}std::cout << std::endl;
	}
				

}

void NeuralNetwork::setInput(std::vector<float> input) {
	this->input = input;
}
void NeuralNetwork::setOutput(std::vector<float> output) {
	this->output = output;
}

void NeuralNetwork::feedforward() {
	neurons[0] = input;

	float sum;

	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < neuronLayers[i + 1]; ++j) {
			sum = 0;
			for (int k = 0; k < neuronLayers[i]; ++k) {
				sum += neurons[i][k] * weights[i][k][j];
			}
			neurons[i + 1][j] = sigmoidFunction(sum);
		}
	}
}

float NeuralNetwork::train() {
	return 0;
}

void NeuralNetwork::randomizeWeights() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-1.f, 1.f);

	for (int i = 0; i < N - 1; ++i)
		for (int j = 0; j < neuronLayers[i]; ++j)
			for (int k = 0; k < neuronLayers[i + 1]; ++k)
				weights[i][j][k] = dist(gen);
}


std::vector<float> NeuralNetwork::getOutput() {
	return neurons[N - 1];
}


float NeuralNetwork::sigmoidFunction(float x) {
	return 1.0f / (1 + pow(2.71828, -x));
}