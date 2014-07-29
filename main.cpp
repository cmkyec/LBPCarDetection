#include <iostream>
#include <iomanip>
#include "utility.h"
#include "LBPCarDetect.h"


int main()
{
	std::string textFilePath("./imageData/negative_train2_path.txt");
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

	std::string featureFilePath("./negative_train_feature_method2.dat");
	writeSVMTrainingData(features, featureFilePath, -1);
	system("pause");
	return 0;
}


/*
int main()
{
	const char* posTxtFile = "./imageData/positive_train_path.txt";
	const char* negTxtFile = "./imageData/negative_train2_path.txt";
	const char* modelFile = "./car_lbp2_method2.model";
	gentech::CLBPCarDetect detector;

	detector.train(posTxtFile, negTxtFile, modelFile);
	system("pause");
	return 0;
}
*/

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
	const char* modelFilePath = "./features/car_lbp2_method2.model";
	float thres = 1.98f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);
	cv::Mat img = cv::imread("./testImages/00036.png");
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
	const char* modelFilePath = "./features/car_lbp2_method2.model";
	float thres = 1.7f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);

	cv::VideoCapture video("./testImages/2014-04-14_15-29-01.avi");
	if (!video.isOpened()) {
		std::cerr << "can not open the video." << std::endl;
		system("pause");
		return 1;
	}
	
	cv::Mat Frame;
	while (true) {
		video >> Frame;
		if (!Frame.data) break;
		std::vector<cv::Rect> carPos;
		detector.detect(Frame, carPos);

		for (std::size_t i = 0; i < carPos.size(); ++i) {
			cv::rectangle(Frame, carPos[i], cv::Scalar(0, 255, 255));
		}
		cv::imshow("show", Frame);
		if (cv::waitKey(1) == 27) break;
	}
	return 0;
}
*/

/*
int main()
{
	cv::Mat img = cv::imread("./testImages/1.png");
	const char* modelFilePath = "./features/car_lbp2_method2.model";
	float thres = 1.f;
	gentech::CLBPCarDetect detector(modelFilePath, thres);

	float v = detector.predict(img);
	std::cout << "v is: " << v << std::endl;
	system("pause");
	return 0;
}
*/