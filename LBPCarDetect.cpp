#include "LBPCarDetect.h"
#include "svm.h"
#include "utility.h"
#include <fstream>

using namespace gentech;

static int g_lookTable[256] = {
	56, 0, 7, 1, 14, 57, 8, 2, 21, 57, 57, 57, 15, 57, 9, 3, 28, 57, 57, 57, 57, 57, 57, 57, 22, 57, 57, 57,
	16, 57, 10, 4, 35, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 29, 57, 57, 57, 57, 57,
	57, 57, 23, 57, 57, 57, 17, 57, 11, 5, 42, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
	57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 36, 57, 57, 57, 57, 57, 57, 57, 57, 57,
	57, 57, 57, 57, 57, 57, 30, 57, 57, 57, 57, 57, 57, 57, 24, 57, 57, 57, 18, 57, 12, 6, 49, 50, 57, 51,
	57, 57, 57, 52, 57, 57, 57, 57, 57, 57, 57, 53, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
	57, 54, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
	57, 57, 57, 57, 57, 57, 57, 55, 43, 44, 57, 45, 57, 57, 57, 46, 57, 57, 57, 57, 57, 57, 57, 47, 57, 57,
	57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 48, 37, 38, 57, 39, 57, 57, 57, 40, 57, 57, 57, 57,
	57, 57, 57, 41, 31, 32, 57, 33, 57, 57, 57, 34, 25, 26, 57, 27, 19, 20, 13, 56
};

static void histFill(cv::Mat& img, cv::Rect rect, float* feature, int len = 58)
{
	cv::Mat tmpImg = img(rect);
	for (int r = 0; r < tmpImg.rows; ++r) {
		for (int c = 0; c < tmpImg.cols; ++c) {
			int v = (int)tmpImg.at<uchar>(r, c);
			feature[v] += 1.0;
		}
	}
	int s = tmpImg.rows * tmpImg.cols;
	for (int i = 0; i < len; ++i) feature[i] /= s;
}

static const int g_featureDim = 522;
static void calcWeight(const char* modelFilePath, std::vector<float>& w)
{
	svm_model* model = svm_load_model(modelFilePath);

	assert(model != NULL);
	assert(model->param.svm_type == C_SVC);
	assert(model->param.kernel_type == LINEAR);

	int sv_count = model->l;
	double* coef = model->sv_coef[0];
	svm_node** nodes = model->SV;

	w.resize(g_featureDim, 0);
	for (int i = 0; i < g_featureDim; ++i) {
		for (int j = 0; j < sv_count; ++j) {
			w[i] += (float)coef[j] * (float)nodes[j][i].value;
		}
	}

	svm_free_model_content(model);
	svm_free_and_destroy_model(&model);
}

static void LBPImage(cv::Mat& src, cv::Mat& lbp)
{
	cv::Mat grayImg;
	if (src.channels() == 3) {
		cv::cvtColor(src, grayImg, CV_BGR2GRAY);
	}
	else {
		src.copyTo(grayImg);
	}

	lbp.create(src.size(), src.depth());
	lbp.setTo(0);
	for (int r = 1; r < grayImg.rows - 1; ++r) {
		const uchar* prev = grayImg.ptr(r - 1);
		const uchar* curr = grayImg.ptr(r);
		const uchar* next = grayImg.ptr(r + 1);
		uchar* pdst = lbp.ptr(r);
		for (int c = 1; c < grayImg.cols - 1; ++c) {
			uchar center = curr[c];
			int value = 0;
			if (curr[c + 1] > center)  value |= 0x01 << 0;
			if (next[c + 1] > center)  value |= 0x01 << 1;
			if (next[c] > center)      value |= 0x01 << 2;
			if (next[c - 1] > center)  value |= 0x01 << 3;
			if (curr[c - 1] > center)  value |= 0x01 << 4;
			if (prev[c - 1] > center)  value |= 0x01 << 5;
			if (prev[c] > center)      value |= 0x01 << 6;
			if (prev[c + 1] > center)  value |= 0x01 << 7;
			pdst[c] = (uchar)g_lookTable[value];
		}
	}
}

static bool rectOverlap(cv::Rect& a, cv::Rect& b)
{
	cv::Point top, bottom;
	top.x = std::max(a.x, b.x);
	top.y = std::max(a.y, b.y);
	bottom.x = std::min(a.x + a.width - 1, b.x + b.width - 1);
	bottom.y = std::min(a.y + a.height - 1, b.y + b.height - 1);

	return (bottom.y - top.y) >= 0 && (bottom.x - top.x) >= 0;
}


CLBPCarDetect::CLBPCarDetect(const char* modelFilePath, float thres)
{
	calcWeight(modelFilePath, m_w);
	m_b = thres;
}

void CLBPCarDetect::computer(cv::Mat& img, std::vector<float>& features)
{
	CV_Assert(img.rows == 48 && img.cols == 48);

	static cv::Mat lbp;
	LBPImage(img, lbp);

	features.resize(58 * 9, 0);
	float* p = &(features[0]);
	// the boundary of the lbp is ignored, actually 46x46 rectangle is used.
	for (int r = 1; r <= 23; r += 11) {
		for (int c = 1; c <= 23; c += 11) {
			cv::Rect rect(c, r, 24, 24);
			histFill(lbp, rect, p);
			p += 58;
		}
	}
}

static int g_window_size = 48;
void CLBPCarDetect::auxiliaryImg(cv::Mat& img, cv::Mat& auImg)
{
	CV_Assert(img.channels() == 1 && img.depth() == CV_8U);

	cv::Mat lbp;
	LBPImage(img, lbp);

	cv::Mat _auImg;
	_auImg.create(lbp.size(), CV_32F);
	_auImg.setTo(0);
	for (int r = 1; r <= _auImg.rows - g_window_size / 2; ++r) {
		for (int c = 1; c <= _auImg.cols - g_window_size / 2; ++c) {
			_auImg.at<float>(r, c) = m_w[(int)lbp.at<uchar>(r, c)] +
				m_w[(int)lbp.at<uchar>(r, c + 11) + 58] +
				m_w[(int)lbp.at<uchar>(r, c + 22) + 116] +
				m_w[(int)lbp.at<uchar>(r + 11, c) + 174] +
				m_w[(int)lbp.at<uchar>(r + 11, c + 11) + 232] +
				m_w[(int)lbp.at<uchar>(r + 11, c + 22) + 290] +
				m_w[(int)lbp.at<uchar>(r + 22, c) + 348] +
				m_w[(int)lbp.at<uchar>(r + 22, c + 11) + 406] +
				m_w[(int)lbp.at<uchar>(r + 22, c + 22) + 464];
		}
	}
	cv::integral(_auImg, auImg, CV_32F);
}

float CLBPCarDetect::predict(cv::Mat& img)
{
	cv::Mat grayImg;
	if (img.channels() == 3) {
		cv::cvtColor(img, grayImg, CV_BGR2GRAY);
	}
	else {
		img.copyTo(grayImg);
	}

	std::vector<float> feature;
	computer(grayImg, feature);
	float v = 0.f;
	for (std::size_t i = 0; i < feature.size(); ++i) {
		v += m_w[i] * feature[i];
	}

	return v;
}

void CLBPCarDetect::rectsMerge(std::vector<cv::Rect>& rawRects, std::vector<cv::Rect>& rects, 
	                       std::vector<float>& scores)
{
	CDisjointSet ds((int)rawRects.size());
	for (std::size_t i = 0; i < rawRects.size(); ++i) {
		for (std::size_t j = i + 1; j < rawRects.size(); ++j) {
			if (rectOverlap(rawRects[i], rawRects[j])) {
				ds.merge((int)i, (int)j);
			}
		}
	}

	std::vector<std::vector<int> > sets;
	ds.subSet(sets);

	rects.clear();
	for (std::size_t i = 0; i < sets.size(); ++i) {
		std::vector<int>& s = sets[i];
		if (s.size() < 2) continue;
		float x, y, width, height;
		x = y = width = height = 0.f;
		float sum = 0.f;
		for (std::size_t j = 0; j < s.size(); ++j) {
			x += scores[s[j]] * rawRects[s[j]].x;
			y += scores[s[j]] * rawRects[s[j]].y;
			width += scores[s[j]] * rawRects[s[j]].width;
			height += scores[s[j]] * rawRects[s[j]].height;
			sum += scores[s[j]];
		}
		cv::Rect aveRect;
		aveRect.x = cvRound(x / sum);
		aveRect.y = cvRound(y / sum);
		aveRect.width = cvRound(width / sum);
		aveRect.height = cvRound(height / sum);
		rects.push_back(aveRect);
	}
}

void CLBPCarDetect::detect(cv::Mat& img, std::vector<cv::Rect>& carPos)
{
	cv::Mat grayImg;
	if (img.channels() == 3) {
		cv::cvtColor(img, grayImg, CV_BGR2GRAY);
	}
	else {
		img.copyTo(grayImg);
	}

	std::vector<cv::Rect> rawCarPos;
	std::vector<float> scores;
	double shrink = 1.0;
	while (grayImg.rows > 100 && grayImg.cols > 100) {
		cv::Mat auImg;
		auxiliaryImg(grayImg, auImg);
		cv::Mat lbp;
		LBPImage(grayImg, lbp);
		for (int r = 12; r < grayImg.rows - 48; r += 24) {
			for (int c = 12; c < grayImg.cols - 48; c += 24) {
				float v = auImg.at<float>(r + 1, c + 1) +
					  auImg.at<float>(r + 25, c + 25) -
					  auImg.at<float>(r + 25, c + 1)  -
					  auImg.at<float>(r + 1, c + 25);
				v = v / 576;
				if (v > m_b) {
					cv::Rect rect(c, r, 48, 48);
					rect.x = cvRound(rect.x / shrink);
					rect.y = cvRound(rect.y / shrink);
					rawCarPos.push_back(rect);
					scores.push_back(v);
				}
			}
		}
		shrink *= 0.85;
		cv::resize(grayImg, grayImg, cv::Size(), shrink, shrink);
	}
	rectsMerge(rawCarPos, carPos, scores);
} 

inline void copyFeatureToNode(std::vector<float>& feature, svm_node* node)
{
	for (int i = 0; i < (int)feature.size(); ++i) {
		node[i].index = i + 1;
		node[i].value = feature[i];
	}
	node[(int)feature.size()].index = -1;
}

void CLBPCarDetect::train(const char* posTxtFile, const char* negTxtFile, const char* modelFile)
{
	std::vector<std::string> posImgsPath, negImgsPath;
	readImagePaths(posTxtFile, posImgsPath);
	readImagePaths(negTxtFile, negImgsPath);
	int nsamples = (int)(posImgsPath.size() + negImgsPath.size());
	svm_node** pnodes = new svm_node*[nsamples];
	double* y = new double[nsamples];
	for (int i = 0; i < (int)posImgsPath.size(); ++i) {
		cv::Mat img = cv::imread(posImgsPath[i]);
		std::vector<float> feature;
		computer(img, feature);
		pnodes[i] = new svm_node[(int)feature.size() + 1];
		copyFeatureToNode(feature, pnodes[i]);
		y[i] = 1;
	}
	for (int i = 0; i < (int)negImgsPath.size(); ++i) {
		cv::Mat img = cv::imread(negImgsPath[i]);
		std::vector<float> feature;
		computer(img, feature);
		pnodes[(int)posImgsPath.size() + i] = new svm_node[(int)feature.size() + 1];
		copyFeatureToNode(feature, pnodes[(int)posImgsPath.size() + i]);
		y[(int)posImgsPath.size() + i] = -1;
	}
	svm_problem problem;
	problem.l = nsamples; problem.x = pnodes; problem.y = y;
	std::cout << "problem is done." << std::endl;

	// all parameters are set to default, except for C and gamma
	// these two parameters are got by grid search
	svm_parameter param;
	param.svm_type = C_SVC; param.kernel_type = LINEAR;
	param.degree = 3; param.gamma = 0.0078125;	
	param.coef0 = 0; param.nu = 0.5; 
	param.cache_size = 100; param.C = 32;
	param.eps = 1e-3; param.p = 0.1;
	param.shrinking = 1; param.probability = 0;
	param.nr_weight = 0; param.weight_label = NULL;
	param.weight = NULL;
	std::cout << "param is done." << std::endl;

	const char* errmsg = svm_check_parameter(&problem, &param);
	if (errmsg) {
		std::cout << "something error: " << errmsg << std::endl;
		return;
	}

	svm_model* model = svm_train(&problem, &param);
	std::cout << "train end." << std::endl;

	int* sv_indices = new int[model->l];
	svm_get_sv_indices(model, sv_indices);
	std::ofstream svIndicesFile("svIndices.txt");
	for (int i = 0; i < model->l; ++i) {
		svIndicesFile << sv_indices[i] << ",";
	}
	svIndicesFile << std::endl;
	svm_save_model(modelFile, model);

	delete[] sv_indices;
	svm_free_model_content(model);
	svm_free_and_destroy_model(&model);
	for (int i = 0; i < nsamples; ++i) delete[] pnodes[i];
	delete[] pnodes;
}