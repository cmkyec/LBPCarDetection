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

	/**
	 * get the LBP feature histogram of the image, the size of the image must be 48x48
	 */
	void computer(cv::Mat& img, std::vector<float>& feature);

	/**
	 * used for test, the image size should be 48x48
	 */
	float predict(cv::Mat& img);

	/**
	 * detect the car position of the image
	 */
	void detect(cv::Mat& img, std::vector<cv::Rect>& carPos);

	/**
	 * train a linear svm model used LBP feature histogram
	 */
	void train(const char* posTxtFile, const char* negTxtFile, const char* modelFile);

	void train(const char* posTxtFile, const char* negTxtFile, double c, double gamma);
protected:
	void CLBPCarDetect::auxiliaryImg(cv::Mat& img, cv::Mat& auImg);

	/**
	 * merge the car positions using the disjoint set algorithm.
	 */
	void rectsMerge(std::vector<cv::Rect>& rawRects, std::vector<cv::Rect>& rects, 
		        std::vector<float>& scores);
private:
	// m_w, m_b are the linear weights and threshold of the svm linear model
	std::vector<float> m_w;
	double m_b;
};

}

#endif /* _LBP_CARDETECT_H_ */