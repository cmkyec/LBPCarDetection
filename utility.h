#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void readImagePaths(std::string textFilePath, 
		    std::vector<std::string>& imgPaths);


void writeSVMTrainingData(std::vector<std::vector<float> >& features,
			  std::string trainFilePath, int flag);

#endif /* _UTILITY_H_ */