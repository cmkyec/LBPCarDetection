#include "utility.h"
#include <cassert>

void readImagePaths(std::string textFilePath,
		    std::vector<std::string>& imgPaths)
{
	imgPaths.resize(0);

	std::ifstream textFile(textFilePath.c_str());
	if (!textFile.is_open()) {
		std::cerr << "can not read the image path text file." << std::endl;
		return;
	}
	std::string path;
	while (std::getline(textFile, path, '\n')) {
		imgPaths.push_back(path);
	}
}

void writeSVMTrainingData(std::vector<std::vector<float> >& features,
			  std::string trainFilePath, int flag)
{
	assert(features.size() != 0);
	assert(flag == 1 || flag == -1);

	std::ofstream trainFile(trainFilePath.c_str());
	if (!trainFile.is_open()) {
		std::cerr << "can not open the training file." << std::endl;
		return;
	}

	for (std::size_t i = 0; i < features.size(); ++i) {
		std::vector<float>& feature = features[i];
		trainFile << flag << " ";
		for (std::size_t j = 0; j < feature.size(); ++j) {
			trainFile << j + 1 << ":" << feature[j] << " ";
		}
		trainFile << std::endl;
	}
}