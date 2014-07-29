#ifndef _LBP_CARDETECT_H_
#define _LBP_CARDETECT_H_

#include <opencv2\opencv.hpp>
#include "disjointSet.h"

namespace gentech
{

class CLBPCarDetect
{
public:
	CLBPCarDetect() {}

	CLBPCarDetect(const char* modelFilePath, float thres);

	~CLBPCarDetect() {}

	void computer(cv::Mat& img, std::vector<float>& feature);

	float predict(cv::Mat& img);

	void detect(cv::Mat& img, std::vector<cv::Rect>& carPos);


	void train(const char* posTxtFile, const char* negTxtFile, const char* modelFile);

protected:
	void CLBPCarDetect::auxiliaryImg(cv::Mat& img, cv::Mat& auImg);

	void rectsMerge(std::vector<cv::Rect>& rawRects, std::vector<cv::Rect>& rects, 
		        std::vector<float>& scores);

private:
	std::vector<float> m_w;
	double m_b;
};

}

#endif /* _LBP_CARDETECT_H_ */