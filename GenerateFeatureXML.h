#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;


class GenerateFeatureXML
{
	string fn_csv;
    vector<Mat> images;
    vector<int> labels;

public:
	void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator);
	void LBPTrainAndSave();
	GenerateFeatureXML(void);
	~GenerateFeatureXML(void);
};

