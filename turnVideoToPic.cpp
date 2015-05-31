#include "ImgMods.h"
#include "turnVideoToPic.h"

//将对应人的录像转变成特征图片
void turnVideoToPic::loadHaarClassifiers()
{
	// Haar Cascade file, used for Face Detection.
	const char *faceCascadeFilename = "cascades\\haarcascade_frontalface_alt.xml";
	const char *noseCascadeFilename = "cascades\\haarcascade_mcs_nose.xml";
	const char *mouthCascadeFilename = "cascades\\haarcascade_mcs_mouth.xml";
	const char *leftEyeCascadeFilename = "cascades\\haarcascade_mcs_lefteye.xml";
	const char *rightEyeCascadeFilename = "cascades\\haarcascade_mcs_righteye.xml";

	//Load the HaarCascade classifier for faces
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0 );
	if( !faceCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", faceCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for noses
	noseCascade = (CvHaarClassifierCascade*)cvLoad(noseCascadeFilename, 0, 0, 0 );
	if( !noseCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", noseCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for mouths
	mouthCascade = (CvHaarClassifierCascade*)cvLoad(mouthCascadeFilename, 0, 0, 0 );
	if( !mouthCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", mouthCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for left eyes
	leftEyeCascade = (CvHaarClassifierCascade*)cvLoad(leftEyeCascadeFilename, 0, 0, 0 );
	if( !leftEyeCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", leftEyeCascadeFilename);
		exit(1);
	}

	//Load the HaarCascade classifier for right eyes
	rightEyeCascade = (CvHaarClassifierCascade*)cvLoad(rightEyeCascadeFilename, 0, 0, 0 );
	if( !rightEyeCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier in '%s'.\n", rightEyeCascadeFilename);
		exit(1);
	}
}

int turnVideoToPic::wipePictureOfBadData(IplImage * faceImage)
{
	IplImage *faceTopLeftImage = 0; //top left face quadrant
	IplImage *faceTopRightImage = 0; //top right face quadrant
	IplImage *faceBottomImage = 0; //bottom half of the face

	//break the face into multiple parts 
	faceTopLeftImage = cropImage(faceImage, cvRect(0, 0, faceImage->width/2, faceImage->height/2)); //top left quadrant
	faceTopRightImage = cropImage(faceImage, cvRect(faceImage->width/2, 0, faceImage->width/2, faceImage->height/2)); // top right quadrant
	faceBottomImage = cropImage(faceImage, cvRect(0, faceImage->height/2, faceImage->width, faceImage->height/2)); //bottom half
	
	CvRect leftEyeRect = cvRect(-1,-1,-1,-1),rightEyeRect = cvRect(-1,-1,-1,-1),noseRect= cvRect(-1,-1,-1,-1),mouthRect= cvRect(-1,-1,-1,-1);
	//find the nose as part of the whole face
	noseRect = detectOneHaarClassifier(faceImage, noseCascade);
	//find the mouth as part of the bottom half
	mouthRect = detectOneHaarClassifier(faceBottomImage, mouthCascade);
	//find the left eye as part of the top left quadrant
	leftEyeRect = detectOneHaarClassifier(faceTopLeftImage, leftEyeCascade);
	//find the right eye as part of tzhe top right quadrant
	rightEyeRect = detectOneHaarClassifier(faceTopRightImage, rightEyeCascade);

	if(leftEyeRect.width >0 && rightEyeRect.width > 0 && noseRect.width > 0 && mouthRect.width > 0)
		return 1;
	return 0;

}

CvRect turnVideoToPic::detectOneHaarClassifier(const IplImage *inputImg, const CvHaarClassifierCascade* cascade )
{
	const CvSize minFeatureSize = cvSize(20, 20);
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_DO_CANNY_PRUNING;	// Only search for 1 face.
	const float search_scale_factor = 1.1f;
	IplImage *detectImg;
	IplImage *greyImg = 0;
	CvMemStorage* storage;
	CvRect mainObjectRectangle;
	double time;
	CvSeq* detectedObjectRectangles;
	int i;

	storage = cvCreateMemStorage(0);
	cvClearMemStorage( storage );

	// If the image is color, use a greyscale copy of the image.
	detectImg = (IplImage*)inputImg;	// Assume the input image is to be used.
	if (inputImg->nChannels > 1) 
	{
		greyImg = cvCreateImage(cvSize(inputImg->width, inputImg->height), IPL_DEPTH_8U, 1 );
		cvCvtColor( inputImg, greyImg, CV_BGR2GRAY );
		detectImg = greyImg;	// Use the greyscale version as the input.
	}

	// Detect all the faces.
	time = (double)cvGetTickCount();
	detectedObjectRectangles = cvHaarDetectObjects( detectImg, (CvHaarClassifierCascade*)cascade, storage,
				search_scale_factor, 3, flags, minFeatureSize );
	time = (double)cvGetTickCount() - time;
	//printf("[Object Detection took %d ms and found %d objects]\n", cvRound( time/((double)cvGetTickFrequency()*1000.0) ), detectedObjectRectangles->total );

	// Get the first detected face (the biggest).
	if (detectedObjectRectangles->total > 0) {
        mainObjectRectangle = *(CvRect*)cvGetSeqElem( detectedObjectRectangles, 0 );
    }
	else
		mainObjectRectangle = cvRect(-1,-1,-1,-1);	// Couldn't find the face.

	//cvReleaseHaarClassifierCascade( &cascade );
	//cvReleaseImage( &detectImg );
	if (greyImg)
		cvReleaseImage( &greyImg );
	cvReleaseMemStorage( &storage );

	return mainObjectRectangle;	// Return the biggest face found, or (-1,-1,-1,-1).
}

IplImage* turnVideoToPic::getIplImageFromFile(char* picFileName)
{
	return new IplImage(picFileName);
}

void processAllCaptures(char* srcFolder)
{
	
}

void turnVideoToPic::extractFace(IplImage *frame)
{
	CvRect faceRect = detectOneHaarClassifier(frame, faceCascade);
	if(faceRect.width>0) 
	{  
		faceImage = cropImage(frame, faceRect); //the face as a whole	

		cvShowImage("眼睛，嘴巴",faceImage);
				
		medianFaceImage =  cvCreateImage(cvGetSize(faceImage),faceImage->depth,faceImage->nChannels);
		src = Mat(faceImage,true);
		medianBlur(src,dest,3);			
		*medianFaceImage = IplImage(dest);
				
	} else
		return;
	cvShowImage("AVI player", frame);//显示当前帧  
	cvWaitKey(10);
	if(medianFaceImage)
		processedFaceImage = equalizeImage(medianFaceImage);
	//椭圆截取
	roi = cvCreateImage(Size(processedFaceImage->width,processedFaceImage->height),8,1);//与运算的前提是两张图片一样大小！
	cvZero(roi);//图片全部置0
	cvAnd(processedFaceImage,processedFaceImage,roi,roi);//进行与运算

	//end
	//增色
	cvEqualizeHist(roi, roi);
			 
	if(wipePictureOfBadData(roi) == 1)
	{
		sprintf(tmpfile,"%s//%d.jpg", AviSavePath, count_tmp/frameNum);//使用帧号作为图片名  
		cvSaveImage(tmpfile,roi); 
	}
}

void turnVideoToPic::getVideoTurnFrayFacePic(int frameNum)
{
	char videoName[256], picName[256];
	char fileName[256],filePic[256];
	cout<<"please input video fileName："<<endl;
	scanf("%s", videoName);
	getchar();
	strcat(videoName,".avi");
	//cout<<"videName"<<videoName<<endl;
	sprintf(fileName,"data\\video\\%s",videoName);
	//cout<<"video name:"<<fileName<<endl;
	cout<<"Please input Picture FileName"<<endl;
	scanf("%s", picName);
	getchar();
	sprintf(filePic,"data\\person\\%s",picName);
	_mkdir(filePic);
	//cout<<"picture name:"<<filePic<<endl;

	IplImage *faceImage = 0; 
	IplImage *processedFaceImage = 0;
	IplImage *medianFaceImage = 0;
	IplImage *roi = 0;

	Mat dest,src,begin,eclip;
	turnVideoToPic::loadHaarClassifiers();

	CvCapture *capture = NULL;  
    IplImage *frame = NULL;  
    char *AviFileName = fileName;  
    char *AviSavePath = filePic;  
    const int jiange = frameNum;//间隔frameNum帧保存一次图片  
    capture = cvCaptureFromAVI(AviFileName);  
    cvNamedWindow("AVI player",1);  
	cvNamedWindow("眼睛，嘴巴",1);
    int count_tmp = 0;//计数 总帧数  
    char tmpfile[100] = {'\0'};  

    while( (frame = cvQueryFrame(capture)) != NULL)  
    { 
		CvRect faceRect;
		CvRect leftEyeRect,rightEyeRect,mouthRect;
		//中值滤波
		
        if (count_tmp % jiange == 0)  
        {  
           
			faceRect = detectOneHaarClassifier(frame, faceCascade);
			if(faceRect.width>0) 
			{  
				/*faceRect.x = faceRect.x + 8*ceil(8*faceRect.width/120.0);
				faceRect.y = faceRect.y + 8*ceil(8*faceRect.width/120.0);
				faceRect.width = faceRect.width - 5;
				faceRect.height = faceRect.height - 5;*/

				faceImage = cropImage(frame, faceRect); //the face as a whole	

				/*leftEyeRect = detectOneHaarClassifier(faceImage,leftEyeCascade);
				rightEyeRect = detectOneHaarClassifier(faceImage,rightEyeCascade);
				mouthRect = detectOneHaarClassifier(faceImage,mouthCascade);

				cvRectangle(faceImage, cvPoint((leftEyeRect).x, (leftEyeRect).y), cvPoint((leftEyeRect).x + (leftEyeRect).width-1, (leftEyeRect).y + (leftEyeRect).height-1), CV_RGB(0,255,0), 1, 8, 0);
				cvRectangle(faceImage, cvPoint((rightEyeRect).x, (rightEyeRect).y), cvPoint((rightEyeRect).x + (rightEyeRect).width-1, (rightEyeRect).y + (rightEyeRect).height-1), CV_RGB(0,255,0), 1, 8, 0);
				cvRectangle(faceImage, cvPoint((mouthRect).x, (mouthRect).y), cvPoint((mouthRect).x + (mouthRect).width-1, (mouthRect).y + (mouthRect).height-1), CV_RGB(0,255,0), 1, 8, 0);
				*/
				cvShowImage("眼睛，嘴巴",faceImage);
				
				medianFaceImage =  cvCreateImage(cvGetSize(faceImage),faceImage->depth,faceImage->nChannels);
				src = Mat(faceImage,true);
				medianBlur(src,dest,3);			
				*medianFaceImage = IplImage(dest);
				
			}else
				continue;
			 cvShowImage("AVI player", frame);//显示当前帧  
			 cvWaitKey(10);
			if(medianFaceImage)
				processedFaceImage = equalizeImage(medianFaceImage);
			
			//椭圆截取
			roi = cvCreateImage(Size(processedFaceImage->width,processedFaceImage->height),8,1);//与运算的前提是两张图片一样大小！
			cvZero(roi);//图片全部置0
			cvEllipse(roi//画椭圆，并填充
				,cvPoint(processedFaceImage->width/2,processedFaceImage->height/2)
				,cvSize(processedFaceImage->width*8/20,processedFaceImage->height/2)
				,0.0
				,0.0
				,360.0
				,CV_RGB(255,255,255)
				,-1,8,0);
			cvAnd(processedFaceImage,processedFaceImage,roi,roi);//进行与运算

			//end
			//增色
			cvEqualizeHist(roi, roi);
			 
			if(wipePictureOfBadData(roi) == 1)
			{
				sprintf(tmpfile,"%s//%d.jpg", AviSavePath, count_tmp/frameNum);//使用帧号作为图片名  
				cvSaveImage(tmpfile,roi); 
			}else
				continue;
        }                 
        if(cvWaitKey(10)>=0) //延时  
        {   break; }  
        ++count_tmp;  
    }  
	if(faceImage)
		cvReleaseImage(&faceImage);
	if(processedFaceImage)
		cvReleaseImage(&processedFaceImage);
	if(roi)
		cvReleaseImage(&roi);
    cvReleaseCapture(&capture);  
    cvDestroyWindow("AVI player");   
	cvDestroyWindow("眼睛，嘴巴");
    std::cout << "总帧数" << count_tmp << std::endl;
}


Mat turnVideoToPic::cropEllipse(Mat &sr)
{
	IplImage *roi;
	IplImage src(sr);
	roi = cvCreateImage(Size(src.width,src.height),8,1);//与运算的前提是两张图片一样大小！
	cvZero(roi);//图片全部置0
	cvEllipse(roi//画椭圆，并填充
		,cvPoint(src.width/2,src.height/2)
		,cvSize(src.width*7/20,src.height/2)
		,0.0
		,0.0
		,360.0
		,CV_RGB(255,255,255)
		,-1,8,0);
	cvAnd(&src,&src,roi,roi);//进行与运算
	Mat reroi1(roi);
	Mat reroi = reroi1.clone();//注意这里一定要用clone方法！释放了roi也就释放了reroi1.
	cvReleaseImage(&roi);
	return reroi;
}
turnVideoToPic::turnVideoToPic(void)
{
}


turnVideoToPic::~turnVideoToPic(void)
{
}
