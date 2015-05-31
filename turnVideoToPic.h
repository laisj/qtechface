#pragma once

#include "opencv/cv.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <direct.h>

using namespace std;

class turnVideoToPic
{
	CvHaarClassifierCascade* faceCascade;
    CvHaarClassifierCascade* noseCascade;
    CvHaarClassifierCascade* mouthCascade;
    CvHaarClassifierCascade* leftEyeCascade;
    CvHaarClassifierCascade* rightEyeCascade;
public:
	Mat cropEllipse(Mat &sr);
	void getVideoTurnFrayFacePic(int);
	int wipePictureOfBadData(IplImage *faceImage);
	void loadHaarClassifiers();
	CvRect detectOneHaarClassifier(const IplImage *inputImg, const CvHaarClassifierCascade* cascade );
	turnVideoToPic(void);
	~turnVideoToPic(void);
};

