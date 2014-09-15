#include <iostream>
#include <iomanip>
#include "utility.h"
#include "LBPCarDetect.h"

/*
int main()
{
	std::string textFilePath("./imageData/huning/positive_train.txt");
	std::vector<std::string> imgPaths;
	readImagePaths(textFilePath, imgPaths);

	std::vector<std::vector<float> > features;
	gentech::CLBPCarDetect detector;
	for (std::size_t i = 0; i < imgPaths.size(); ++i) {
		std::cout<<std::setw(2)<< (int)(i * 1.0 / imgPaths.size() * 100) << "%";
		cv::Mat img = cv::imread(imgPaths[i]);
		std::vector<float> feature;
		detector.computer(img, feature);
		features.push_back(feature);
		std::cout << "\b\b\b";
	}
	std::cout << "100%" << std::endl;

	std::string featureFilePath("./features/huning/positive_train_feature.dat");
	writeSVMTrainingData(features, featureFilePath, 1);
	system("pause");
	return 0;
}
*/

/*
int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cout << "extractFeatureData.exe text_file_of_imagess_path text_file_of_feature_data positive_or_negative_flag" << std::endl;
		std::cout << "positive_or_negative_flag must be 1 or -1, 1 respond for positive and -1 respond for negative" << std::endl;
		return 1;
	}
	//std::string textFilePath("./imageData/huning/positive_train.txt");
	std::string textFilePath(argv[1]);
	std::vector<std::string> imgPaths;
	readImagePaths(textFilePath, imgPaths);

	std::vector<std::vector<float> > features;
	gentech::CLBPCarDetect detector;
	for (std::size_t i = 0; i < imgPaths.size(); ++i) {
		std::cout << std::setw(2) << (int)(i * 1.0 / imgPaths.size() * 100) << "%";
		cv::Mat img = cv::imread(imgPaths[i]);
		std::vector<float> feature;
		detector.computer(img, feature);
		features.push_back(feature);
		std::cout << "\b\b\b";
	}
	std::cout << "100%" << std::endl;

	//std::string featureFilePath("./features/huning/positive_train_feature.dat");
	std::string featureFilePath(argv[2]);
	int flag = atoi(argv[3]);
	writeSVMTrainingData(features, featureFilePath, flag);
	system("pause");
	return 0;
}
*/

/*
int main()
{
	const char* posTxtFile = "./imageData/huning/positive_train.txt";
	const char* negTxtFile = "./imageData/huning/negative_train.txt";
	const char* modelFile = "./huning.model";
	gentech::CLBPCarDetect detector;

	detector.train(posTxtFile, negTxtFile, modelFile);
	system("pause");
	return 0;
}
*/


int main(int argc, char** argv)
{
	if (argc < 5) {
		std::cout << "CarDetectorTrain.exe path_to_positive_images path_to_negative_images c_value, gamma_value" << std::endl;
		return -1;
	}
	std::string posTxtFile(argv[1]);
	std::string negTextFile(argv[2]);
	double c = atof(argv[3]);
	double gamma = atof(argv[4]);
	gentech::CLBPCarDetect detector;
	detector.train(posTxtFile.c_str(), negTextFile.c_str(), c, gamma);
	return 0;
}


/*
int main()
{
	const char* modelFilePath = "./features/method_a_model.dat";
	float thres = 2.85f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);

	//std::string txtFilePath("./testImageData/positive_test_path.txt");
	std::string txtFilePath("./testImageData/negative_test3_path.txt");
	std::vector<std::string> imgPaths;
	readImagePaths(txtFilePath, imgPaths);

	int count = 0;
	for (std::size_t i = 0; i < imgPaths.size(); ++i) {
		std::cout << std::setw(5) << i + 1;
		cv::Mat img = cv::imread(imgPaths[i]);
		float v = detector.predict(img);
		if (v > thres) count++;
		std::cout << "\b\b\b\b\b";
	}
	std::cout << "done." << std::endl;
	std::cout << "count / total: " << count << "/" << imgPaths.size() << std::endl;
	system("pause");
	return 0;
}
*/

/*
int main()
{
	const char* modelFilePath = "./features/car_lbp.model";
	float thres = 2.7f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);

	std::string textFilePath("./imageData/neg_image_origin_path.txt");
	std::vector<std::string> imgPaths;
	readImagePaths(textFilePath, imgPaths);

	char savePath[200];
	memset(savePath, 0, sizeof(savePath));
	const char* saveFolder = ".\\imageData\\negative_train_3";
	int count = 0;
	for (int i = 0; i < (int)imgPaths.size(); ++i) {
		cv::Mat img = cv::imread(imgPaths[i]);
		std::vector<cv::Rect> carPos;
		detector.detect(img, carPos);
		for (std::size_t j = 0; j < carPos.size(); ++j) {
			cv::Rect& rect = carPos[j];
			cv::Mat imgTmp = img(rect);
			sprintf_s(savePath, sizeof(savePath), "%s\\%05d.png", saveFolder, count++);
			cv::imwrite(savePath, imgTmp);
		}
		std::cout << "process: " << i + 1 << std::endl;
	}
	system("pause");
	return 0;
}
*/


/*
int main()
{
	const char* modelFilePath = "./features/huning/huning.model";
	float thres = 2.f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);
	cv::Mat img = cv::imread("./testImages/debugTmp/762.jpg");
	std::vector<cv::Rect> carPos;
	detector.detect(img, carPos);
	for (std::size_t i = 0; i < carPos.size(); ++i) {
		cv::rectangle(img, carPos[i], cv::Scalar(0, 255, 255));
	}
	cv::imshow("show", img);
	cv::waitKey(0);

	return 0;
}
*/

/*
int main()
{
	const char* modelFilePath = "./features/huning/huning.model";
	float thres = 1.f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);

	cv::VideoCapture video("./testImages/data_scheme20120503202002000002302");
	if (!video.isOpened()) {
		std::cerr << "can not open the video." << std::endl;
		system("pause");
		return 1;
	}
	
	cv::Mat Frame;
	//char savepath[200];
	//memset(savepath, 0, sizeof(savepath));
	//int index = 1;
	while (true) {
		video >> Frame;
		if (!Frame.data) break;
		cv::Rect rect(0, Frame.rows / 4, Frame.cols, Frame.rows * 3 / 4);
		cv::Mat img = Frame(rect);
		std::vector<cv::Rect> carPos;
		detector.detect(img, carPos);

		for (std::size_t i = 0; i < carPos.size(); ++i) {
			cv::rectangle(img, carPos[i], cv::Scalar(0, 255, 255));
		}
		//sprintf_s(savepath, sizeof(savepath), "./test/%05d.png", index++);
		//cv::imwrite(savepath, Frame);
		cv::imshow("show", img);
		if (cv::waitKey(1) == 27) break;
	}
	return 0;
}
*/

/*
int main()
{
	cv::Mat img = cv::imread("./testImages/00013_neg.png");
	const char* modelFilePath = "./features/huning/huning.model";
	float thres = -1.f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);

	float v = detector.predict(img);
	std::cout << "v is: " << v << std::endl;
	system("pause");
	return 0;
}
*/