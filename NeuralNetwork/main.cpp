#include <iostream>
#include <random>
#include "NeuralNetwork.h"

int main()
{
	try {

		std::vector<int> v = { 2, 4, 4, 3, 1 };

		NeuralNetwork net(v);
		net.setLearningRate(0.353423);

		std::cout << "Error: " << net.getError() << std::endl;
		while (net.getError() > 0.01){

			std::cout << "Error: " << net.getError() << std::endl;


			/*net.setInput({ 1, 0 });
			net.setOutput({ 1 });

			net.feedforward();
			net.train();*/

			////

			net.setInput({ 1, 1 });
			net.setOutput({ 1 });

			net.feedforward();
			net.train();

			////

			/*net.setInput({ 0, 1 });
			net.setOutput({ 1 });

			net.feedforward();
			net.train();*/

			////

			net.setInput({ 0, 0 });
			net.setOutput({ 0 });

			net.feedforward();
			net.train();
		}


		{
			net.setInput({ 0, 0 });
			net.feedforward();

			net.printOut();
		}
		{
			net.setInput({ 0, 1 });
			net.feedforward();

			net.printOut();
		}
		{
			net.setInput({ 1, 0 });
			net.feedforward();

			net.printOut();
		}
		{
			net.setInput({ 1, 1 });
			net.feedforward();

			net.printOut();
		}
	}
	catch (std::runtime_error &err) {
		std::cerr << err.what() << std::endl;
	}

	

	return 0;
}