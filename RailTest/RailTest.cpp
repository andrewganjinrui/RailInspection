// RailTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <opencv.hpp>
#include <vector>  
#include <io.h> 

using namespace std;
using namespace cv;

int bandwidth = 3;
float gaussianIndex = 1.0 / sqrtf(6.18);
int iternum = 10;
int clusternum = 20;
float thr_segment = 0.02; //This parameter is very important 


void getFiles(string path, vector<string>& files, vector<string> &ownname)
{
	/*files存储文件的路径及名称(eg.   C:\Users\WUQP\Desktop\test_devided\data1.txt)
	ownname只存储文件的名称(eg.     data1.txt)*/

	//文件句柄  
	intptr_t   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{  /*
			   if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
			   getFiles( p.assign(path).append("\\").append(fileinfo.name), files, ownname ); */
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				ownname.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void mean_shift(Mat& imgMat, Mat& clustersMat, Mat& weightsMat, Mat& labelMat)
{
	int imgrows = imgMat.rows;
	int imgcols = imgMat.cols;
	int repoptsnum = imgrows / 4;
	Mat roughClusterMat = Mat::zeros(repoptsnum, imgcols, CV_32F);
	RNG rng;
	int ii = 0, jj = 0, choseRep;
	Mat repMat, extendMat;
	Mat upMat, downMat, outMat;

	while (ii < repoptsnum)
	{
		choseRep = int(rng.uniform(0, imgrows));
		repMat = imgMat.row(choseRep).clone();
		while (jj++ < iternum){
			extendMat = repeat(repMat, imgrows, 1);
			extendMat = (extendMat - imgMat) / bandwidth;
			extendMat = (extendMat.mul(extendMat)) * (-0.5);
			exp(extendMat, extendMat);
			extendMat = extendMat * gaussianIndex;

			reduce(extendMat, downMat, 0, CV_REDUCE_SUM);
			reduce(extendMat.mul(imgMat), upMat, 0, CV_REDUCE_SUM);

			repMat = upMat / downMat;
		}
		repMat.copyTo(roughClusterMat.row(ii));
		ii++;
	}
	cv::sort(roughClusterMat, roughClusterMat, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	
	roughClusterMat.row(0).copyTo(clustersMat);
	int intergap = repoptsnum / (clusternum - 1);
	for (ii = 1; ii < clusternum-1; ii++)
	{
		clustersMat.push_back(roughClusterMat.row(ii*intergap));
	}
	clustersMat.push_back(roughClusterMat.row(repoptsnum - 1));

	weightsMat = Mat::zeros(clustersMat.size(), CV_32F);
	labelMat = Mat::zeros(imgMat.size(), CV_8U);
	Mat diffMat, sortidxMat, countMat, tempMat;
	Point minLoc;
	for (ii = 0; ii < imgcols; ii++)
	{
		diffMat = abs(repeat(imgMat.col(ii), 1, clusternum) - repeat(clustersMat.col(ii).t(), imgrows, 1));
		sortIdx(diffMat, sortidxMat, SORT_EVERY_ROW + SORT_ASCENDING);

		for (jj = 0; jj < imgrows; jj++)
		{
			minMaxLoc(sortidxMat.row(jj), 0, 0, &minLoc, 0);
			if (minLoc.x < clusternum/5)
				labelMat.at<uchar>(jj, ii) = 1;
		}

		tempMat = (sortidxMat < 1) / 255;
		tempMat.convertTo(tempMat, CV_32F);
		reduce(tempMat, countMat, 0, CV_REDUCE_SUM);
		
		countMat = countMat / imgrows;
		tempMat = countMat.t();
		tempMat.copyTo(weightsMat.col(ii));
	}
	tempMat = (weightsMat > 0.1) / 255;
	tempMat.convertTo(tempMat, CV_32F);
	weightsMat = weightsMat.mul(tempMat);
}

void log_normalization(Mat& Img)
{
	Img = (Img + 1)*0.5;
	Mat logImg;
	log(Img, logImg);

	Mat tmp_m, tmp_std;
	double meanV, stdV;
	meanStdDev(logImg, tmp_m, tmp_std);
	meanV = tmp_m.at<double>(0, 0);
	stdV = tmp_std.at<double>(0, 0);

	logImg = (logImg - meanV) / stdV;
	normalize(logImg, Img, 1.0, 0.0, NORM_MINMAX);
}

int _tmain(int argc, _TCHAR* argv[])
{ 
	std::vector< std::string > xlm_list;
	std::vector< std::string > ownname_list;
	//储存大文件的目录路径  
	std::string FLAGS_xml_dir = "D:\\My Projects\\RailTest\\defect";
	getFiles(FLAGS_xml_dir, xlm_list, ownname_list);

	double t0 = (double)getTickCount();
	int k = 1;
	Mat img;
	for (k= 0; k < xlm_list.size(); k++)
	{
		img = imread(xlm_list[k], CV_LOAD_IMAGE_UNCHANGED);
		if (img.empty())
			return -1;
		if (img.channels() != 1)
		{
			cvtColor(img, img, CV_BGR2GRAY);
		}
		if (img.depth() == CV_8U)
		{
			img.convertTo(img, CV_32F);
		}
		log_normalization(img);

		int m_rows = img.rows;
		int m_cols = img.cols;

		Mat LocationPrior(1, m_cols, CV_32F);
		float* lpData = (float*)LocationPrior.data;
		float tempV;
		for (int i = 0; i < m_cols; i++)
		{
			tempV = (float)i / (float)(m_cols - 1);
			*lpData = 0.4 * tempV * (1 - tempV);
			lpData++;
		}
		Mat LocationPriorMat;
		repeat(LocationPrior, m_rows, 1, LocationPriorMat);

		Mat clustersMat, weightsMat, labelMat;
		mean_shift(img, clustersMat, weightsMat, labelMat);

		Mat tempMat;
		int ii, jj;
		Mat diffMat[3];
		Mat saliencyMat = Mat::zeros(img.size(), CV_32F);
		for (ii = 0; ii < m_cols; ii++)
		{
			jj = max(ii - 1, 0);
			diffMat[0] = abs(repeat(img.col(ii), 1, clusternum) - repeat(clustersMat.col(jj).t(), m_rows, 1));
			diffMat[0] = diffMat[0].mul(repeat(weightsMat.col(jj).t(), m_rows, 1));

			jj = ii;
			diffMat[1] = abs(repeat(img.col(ii), 1, clusternum) - repeat(clustersMat.col(jj).t(), m_rows, 1));
			diffMat[1] = diffMat[0].mul(repeat(weightsMat.col(jj).t(), m_rows, 1));

			jj = min(ii + 1, m_cols - 1);
			diffMat[2] = abs(repeat(img.col(ii), 1, clusternum) - repeat(clustersMat.col(jj).t(), m_rows, 1));
			diffMat[2] = diffMat[0].mul(repeat(weightsMat.col(jj).t(), m_rows, 1));

			tempMat = (diffMat[0] + diffMat[1] + diffMat[2]) / 3.0;
			reduce(tempMat, saliencyMat.col(ii), 1, CV_REDUCE_SUM);
		}
		saliencyMat = saliencyMat.mul(LocationPriorMat);

		tempMat = (saliencyMat > thr_segment) / 255;
		labelMat = labelMat.mul(tempMat);
		//保存结果图
		imwrite("D:\\My Projects\\RailTest\\defectResult\\" + ownname_list[k], labelMat*255);
	}

	t0 = (double)getTickCount() - t0;
	cout << " time:\n " << t0 * 1000 / getTickFrequency() / k << " ms" << endl;

	return 0;
}

