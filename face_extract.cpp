// enter point

#include <iostream>
#include <string.h>
#include "VideoCap.h"
#include "turnVideoToPic.h"
#include "GenerateFeatureXML.h"
#include "GenerateCSV.h"
using namespace std;
using namespace cv;

static void help()
{
	cout<<"It's help!"<<endl;
	cout<<"2: Convert video to picture"<<endl;
	cout<<"q: quit the test system!"<<endl;
}

int main(int argc, const char** argv)
{
	char n;
	char num[256];
	VideoCap videoCap;
	turnVideoToPic turnVideo;
	GenerateFeatureXML generateFeature;
	GenerateCSV generateCsv;
	while(true)
	{
		help();
		n = '0';
		if (scanf("%s", num) <= 0)
		{
			perror("scanf");
			exit(-1);
		}
		getchar();
		if (strlen(num) > 2)
		{
			goto reset;
		}
		n = num[0];

		switch( n )
		{
		case '1':
			videoCap.recordVideo();		
			break;
		case '2':
			int frameNum;
			cout<<"please input frame number! the number is between 5 and 10 "<<endl;
			scanf("%d",&frameNum);
			if( frameNum<=0 || frameNum >= 10)
				exit(-1);
			getchar();
			turnVideo.getVideoTurnFrayFacePic(frameNum);
			break;
		case '3':
			generateCsv.FindAllFile("data\\person");
			break;
		case '4':
			generateFeature.LBPTrainAndSave();
			break;
		case 'q':
			videoCap.~VideoCap();
			exit(0);
		default:
			break;
		}
reset:
		cout<<"input error!"<<endl;
	}
	return 0;
}

