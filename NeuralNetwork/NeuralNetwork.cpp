#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(){

}
NeuralNetwork::NeuralNetwork(std::vector<int> &neuronLayers) {
	init(neuronLayers);
	randomizeWeights();		
}

void NeuralNetwork::init(std::vector<int> &neuronLayers) {
	N = neuronLayers.size();
	this->neuronLayers = neuronLayers;

	neurons.resize(N);
	weights.resize(N - 1);

	for (int i = 0; i < N; ++i)
		neurons[i].resize(neuronLayers[i], 1);

	for (int i = 0; i < N - 1; ++i) {
		weights[i].resize(neuronLayers[i]);
		for (int j = 0; j < neuronLayers[i]; ++j) {
			weights[i][j].resize(neuronLayers[i + 1]);
		}
	}
}

void NeuralNetwork::setInput(std::vector<float> input) {
	this->input = input;
}
void NeuralNetwork::setOutput(std::vector<float> output) {
	this->output = output;
}
void NeuralNetwork::setLearningRate(float learningRate) {
	this->learningRate = learningRate;
}

void NeuralNetwork::feedforward() {
	if (neurons[0].size() != input.size())
		throw std::runtime_error("ERROR::INCORRECT_INPUT");
	
	neurons[0] = input;
	

	float sum;

	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < neuronLayers[i + 1]; ++j) {
			sum = 0;
			for (int k = 0; k < neuronLayers[i]; ++k) {
				sum += neurons[i][k] * weights[i][k][j];
			}
			neurons[i + 1][j] = sigmoid(sum);
		}
	}
}

float NeuralNetwork::train() /*Gradient descent*/ {
	if (neurons[N - 1].size() != output.size())
		throw std::runtime_error("ERROR::INCORRECT_OUTPUT");

	float Error = 0;
	for (int i = 0; i < neurons[N - 1].size(); ++i)
		Error += (neurons[N - 1][i] - output[i]) * (neurons[N - 1][i] - output[i]);
	Error /= neurons[N - 1].size();

	deltaNeurons = neurons;

	
	for (int i = 0; i < neuronLayers[N - 1]; ++i)
		deltaNeurons[N - 1][i] = output[i] - neurons[N - 1][i];
	
	for (int k = N - 2; k > 0; --k) {
		for (int i = 0; i < neuronLayers[k]; ++i) { // N - 2
			deltaNeurons[k][i] = 0;
			for (int j = 0; j < neuronLayers[k + 1]; ++j) { // N - 1
				deltaNeurons[k][i] += deltaNeurons[k + 1][j] * weights[k][i][j];
			}
		}
	}

	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < neuronLayers[i]; ++j) {
			for (int k = 0; k < neuronLayers[i + 1]; ++k) {
				weights[i][j][k] += deltaNeurons[i + 1][k] * sigmoidDerivative(neurons[i + 1][k]) * neurons[i][j] * learningRate;
			}
		}
	}
	return Error;
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

void NeuralNetwork::loadWeightsFromTxtFile(const std::string &filename) {
	std::string name = filename + ".txt";
	std::ifstream file(name);


	if (!file.is_open())
		throw std::runtime_error("ERROR::FILE_OPEN");

	int n;
	file >> n;

	std::vector<int> neuronLayers(n, 0);
	for (int i = 0; i < n; ++i)
		file >> neuronLayers[i];

	weights.clear();
	neurons.clear();

	init(neuronLayers);

	for (int i = 0; i < N - 1; ++i)
		for (int j = 0; j < neuronLayers[i]; ++j)
			for (int k = 0; k < neuronLayers[i + 1]; ++k)
				file >> weights[i][j][k];

	file.close();
}

void NeuralNetwork::saveWeightsToTxtFile(const std::string &filename) {
	std::string name = filename + ".txt";
	std::ofstream file;
	file.open(name, std::ofstream::out | std::fstream::trunc);

	if (!file.is_open())
		std::cerr << "problem to open file for saving" << std::endl;

	file << N << '\n';
	for (int i = 0; i < N; ++i)
		file << neuronLayers[i] << ' ';
	file << "\n\n";

	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < neuronLayers[i]; ++j) {
			for (int k = 0; k < neuronLayers[i + 1]; ++k) {
				file << weights[i][j][k] << ' ';
			}file << '\n';
		}file << '\n';
	}
}


std::vector<float> NeuralNetwork::getOutput() const {
	return neurons[N - 1];
}

float NeuralNetwork::getError() {
	if (neurons[N - 1].size() != output.size())
		return 100;

	float Error = 0;
	for (int i = 0; i < neurons[N - 1].size(); ++i)
		Error += (neurons[N - 1][i] - output[i]) * (neurons[N - 1][i] - output[i]);
	Error /= neurons[N - 1].size();
	return Error;
}

float NeuralNetwork::sigmoid(float x) const {
	return 1.0f / (1 + pow(2.71828, -x));
}

float NeuralNetwork::sigmoidDerivative(float x) const {
	return sigmoid(x) * (1 - sigmoid(x));
}



void NeuralNetwork::printNeurons() const {
	std::cout << " === " << std::endl;
	std::cout << "Neurons:" << std::endl;
	for (int i = 0; i < neurons.size(); ++i) {
		for (int j = 0; j < neurons[i].size(); ++j) {
			std::cout << neurons[i][j] << " ";
		}std::cout << std::endl << std::endl;
	}
	std::cout << " === " << std::endl;
}
void NeuralNetwork::printInput() const {
	std::cout << " === " << std::endl;
	std::cout << "Input:" << std::endl;
	for (int j = 0; j < neurons[0].size(); ++j) {
		std::cout << neurons[0][j] << " ";
	}
	std::cout << " === " << std::endl;
}
void NeuralNetwork::printOut() const {
	std::cout << " === " << std::endl;
	std::cout << "Out:" << std::endl;
	for (int j = 0; j < neurons[N - 1].size(); ++j) {
		std::cout << neurons[N - 1][j] << " ";
	}
	std::cout << " === " << std::endl;
}
void NeuralNetwork::printWeights() const {
	std::cout << " === " << std::endl;
	std::cout << "Weights:" << std::endl;
	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < neuronLayers[i]; ++j) {
			for (int k = 0; k < neuronLayers[i + 1]; ++k) {
				std::cout << weights[i][j][k] << " ";
			}std::cout << std::endl;
		}std::cout << std::endl;
	}
	std::cout << " === " << std::endl;
}