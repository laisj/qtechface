#pragma once
#include <io.h>
#include <iostream>
#include <vector>
#include <direct.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
using namespace std;
class GenerateCSV
{
	int label;
	FILE * fd;
	string name;
public:
	vector<string> files;
	void writeToCsv();
	void browseDir(const char *dir);
	void FindAllFile(string path);
	void getFilesAll(string path, vector<string>& files,FILE* fWrite);
	GenerateCSV(void);
	~GenerateCSV(void);
};

