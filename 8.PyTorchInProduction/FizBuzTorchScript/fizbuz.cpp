// ~/myp/HOD/8.P/FizBuzTorchScript/build> ./fizbuz ../fizbuz_model.pt 2

#include <torch/script.h>

#include <iostream>
#include <memory>
#include <string>

int main(int argc, const char* argv[]) {
	if (argc != 3) {
		std::cerr << "usage: <appname> <path> <int>\n";
		return -1;
	}
	std::string arg = argv[2];
	int x = std::stoi(arg);
	float array[10];

	int i;
	int j = 9;
	for (i = 0; i < 10; ++i) {
	    array[j] = (x >> i) & 1;
	    j--;
	}
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor tensor_in = torch::from_blob(array, {1, 10}, options);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(tensor_in);
	std::cout << inputs << '\n';
	/*
		1e-45 *
			 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.4013  0.0000
			[ Variable[CPUFloatType]{1,10} ]
	*/


	at::Tensor output = module->forward(inputs).toTensor();
	std::cout << output << '\n';

	/*
		 3.7295 -23.8977 -8.2652 -1.3901
			[ Variable[CPUFloatType]{1,4} ]
	*/
}