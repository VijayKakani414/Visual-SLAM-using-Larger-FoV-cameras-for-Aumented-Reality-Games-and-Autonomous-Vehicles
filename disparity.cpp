#pragma once

#include < stdio.h>  
#include < opencv2\opencv.hpp>

#include "time.h"
#include < vector> 

#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace std;
using namespace cv;

void convertMat2Vec(Mat& imagepoint, vector<Point2f>& vImagePoint){
	vImagePoint.resize(imagepoint.rows);
	for (int i = 0; i < imagepoint.rows; i++){
		Point2f& point2f = vImagePoint[i];
		point2f.x = imagepoint.at<float>(i, 0);
		point2f.y = imagepoint.at<float>(i, 1);
	}
}

void drawMatchingPoints2(Mat& canvas, const Mat& imagePoints1, const Mat& imagePoints2){
	//draw lines and circles
	int rows, cols;
	RNG rng(0xFFFFFFFF);
	rows = canvas.rows;	cols = canvas.cols / 2;
	for (int i = 0; i < imagePoints1.rows; i++){
		Point2f p1, p2;
		p1 = Point2f((float)imagePoints1.at<float>(i, 0), (float)imagePoints1.at<float>(i, 1));
		p2 = Point2f((float)imagePoints2.at<float>(i, 0) + cols, (float)imagePoints2.at<float>(i, 1));
		circle(canvas, p1, 2, Scalar(0, 0, 255));
		circle(canvas, p2, 2, Scalar(0, 0, 255));
		line(canvas, p1, p2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));

	}
}

void drawMatchingPoints2(Mat& canvas, const vector<Point2f>& imagePoints1, const vector<Point2f>& imagePoints2){
	//draw lines and circles
	int rows, cols;
	RNG rng(0xFFFFFFFF);
	rows = canvas.rows;	cols = canvas.cols / 2;
	for (int i = 0; i < imagePoints1.size(); i++){
		Point2f p1, p2;
		p1 = imagePoints1[i];
		p2 = imagePoints2[i]; p2.x += cols;
		circle(canvas, p1, 2, Scalar(0, 0, 255));
		circle(canvas, p2, 2, Scalar(0, 0, 255));
		line(canvas, p1, p2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));

	}
}

void detectCorrespondence(Mat& img1, Mat& img2, vector<Point2f>& corner1, vector<Point2f>& corner2, int nPtMax);
void disparity2depth(Mat& mDisparity, Mat& mDepth, float fFocal, Mat& tvec1, Mat& tvec2);

void main1(){

	//global variables
	Mat img[2], imgC1[2];
	Mat imagepoint1, imagepoint2, worldpoint;
	vector<Point2f> vImagePoint1, vImagePoint2;
	Mat intrinsic, distcoeff, tvec1, rvec1, tvec2, rvec2;
	Mat map1, map2, rmat1, rmat2;
	Mat imgpts[2], imgpts_n[2];
	vector<Vec3f> lines[2];
	Mat R, T, E, F;
	Size imgSize;
	Mat disparity, disparity8;


	img[0] = imread("ExpEssentialMatrix\\1image.jpg");
	img[1] = imread("ExpEssentialMatrix\\2image.jpg");

	cvtColor(img[0], imgC1[0], CV_BGR2GRAY);
	cvtColor(img[1], imgC1[1], CV_BGR2GRAY);

	imgSize = img[0].size();
	//read image points
	cv::FileStorage fs;
	string st;	

	fs = FileStorage("ExpEssentialMatrix\\1imagePoints.yml", cv::FileStorage::READ);
	fs["image points"] >> imagepoint1;
	fs = FileStorage("ExpEssentialMatrix\\2imagePoints.yml", cv::FileStorage::READ);
	fs["image points"] >> imagepoint2;
	fs = FileStorage("ExpEssentialMatrix\\1worldPoints.yml", cv::FileStorage::READ);
	fs["world points"] >> worldpoint;
	fs = FileStorage("ExpEssentialMatrix\\IntrinsicMatrix.yml", cv::FileStorage::READ);
	fs["Intrinsic matrix"] >> intrinsic;
	fs = FileStorage("ExpEssentialMatrix\\DistortionCoeffs.yml", cv::FileStorage::READ);
	fs["Distortion coeffs matrix"] >> distcoeff;
	fs = FileStorage("ExpEssentialMatrix\\1rotation.yml", cv::FileStorage::READ);
	fs["rotation vector"] >> rvec1;
	fs = FileStorage("ExpEssentialMatrix\\1translation.yml", cv::FileStorage::READ);
	fs["translation vector"] >> tvec1;
	fs = FileStorage("ExpEssentialMatrix\\2rotation.yml", cv::FileStorage::READ);
	fs["rotation vector"] >> rvec2;
	fs = FileStorage("ExpEssentialMatrix\\2translation.yml", cv::FileStorage::READ);
	fs["translation vector"] >> tvec2;
	fs.release();

	//convert to the float type
	imagepoint1.convertTo(imagepoint1, CV_32FC1);
	imagepoint2.convertTo(imagepoint2, CV_32FC1);
	worldpoint.convertTo(worldpoint, CV_32FC1);
	//intrinsic.convertTo(intrinsic, CV_32FC1);
	rvec1.convertTo(rvec1, CV_32FC1);
	tvec1.convertTo(tvec1, CV_32FC1);
	rvec2.convertTo(rvec2, CV_32FC1);
	tvec2.convertTo(tvec2, CV_32FC1);

	//convert matrix to vector
	convertMat2Vec(imagepoint1, vImagePoint1);
	convertMat2Vec(imagepoint2, vImagePoint2);

	detectCorrespondence(imgC1[0], imgC1[1], vImagePoint1, vImagePoint2, 50);

	imgpts[0] = Mat(vImagePoint1);
	imgpts[1] = Mat(vImagePoint2);

	//convert rotation vector to rotation matrix
	Rodrigues(rvec1, rmat1);
	Rodrigues(rvec2, rmat2);


	//undistort the image points
	undistortPoints(imgpts[0], imgpts[0], intrinsic, distcoeff, Mat(), intrinsic);
	undistortPoints(imgpts[1], imgpts[1], intrinsic, distcoeff, Mat(), intrinsic);
	Mat R1, R2, P1, P2, Q;
	

	/*triangulatePoints(proj1, proj2, imgpts[0], imgpts[1], points4D);
	convertPointsFromHomogeneous(points4D.t(), points3D);*/

	//Harley's method
	//find the fundamental matrix F
	//compute the corresponding epipoles
	F = findFundamentalMat(imgpts[0], imgpts[1], FM_RANSAC, 0, 0);
	
	
	Mat H1, H2;
	stereoRectifyUncalibrated(imgpts[0], imgpts[1], F, imgSize, H1, H2);
	R1 = intrinsic.inv()*H1*intrinsic;
	R2 = intrinsic.inv()*H2*intrinsic;
	P1 = intrinsic;	P2 = intrinsic;
	



	Mat rmap[2][2];
	//precompute maps for remap
	initUndistortRectifyMap(intrinsic, distcoeff, R1, P1, imgSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(intrinsic, distcoeff, R2, P2, imgSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	Mat canvas, canvas1;
	double sf;
	int w, h;
	sf = 600.0 / max(imgSize.width, imgSize.height);
	w = round(imgSize.width*sf);
	h = round(imgSize.height*sf);
	canvas.create(h, w * 2, img[0].type());
	Mat rimg, cimg;
	Mat imgGray[2];
	for (int k = 0; k < 2; k++){
		remap(img[k], rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
		cvtColor(rimg, imgGray[k], CV_RGB2GRAY);
		Mat canvasPart = canvas(Rect(k*w, 0, w, h));
		resize(rimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
	}
	RNG rng(0xFFFFFFFF);
	for (int i = 0; i < canvas.rows; i+=16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), 1, 8);


	//compute the disparity
	StereoBM stereoBM(StereoBM::BASIC_PRESET, 112, 17);
	StereoBM sbm;
	sbm.state->SADWindowSize = 9;
	sbm.state->numberOfDisparities = 64;
	sbm.state->preFilterSize = 5;
	sbm.state->preFilterCap = 61;
	sbm.state->minDisparity = -39;
	sbm.state->textureThreshold = 507;
	sbm.state->uniquenessRatio = 0;
	sbm.state->speckleWindowSize = 0;
	sbm.state->speckleRange = 8;
	sbm.state->disp12MaxDiff = 1;

	StereoSGBM sgbm;
	sgbm.SADWindowSize = 15;
	sgbm.numberOfDisparities = 64;
	sgbm.preFilterCap = 4;
	sgbm.minDisparity = -39;
	sgbm.uniquenessRatio = 1;
	sgbm.speckleWindowSize = 150;
	sgbm.speckleRange = 2;
	sgbm.disp12MaxDiff = 10;
	sgbm.fullDP = false;
	sgbm.P1 = 600;
	sgbm.P2 = 2400;



	sbm(imgGray[0], imgGray[1], disparity);
	//sgbm(imgGray[0], imgGray[1], disparity);
	filterSpeckles(disparity, -39, 500, 40);
	normalize(disparity, disparity8, 0, 255, CV_MINMAX, CV_8U);
	


	//draw a canvas for a original images
	canvas1 = Mat::zeros(Size(2 * imgSize.width, imgSize.height), CV_8UC3);
	img[0].copyTo(canvas1.colRange(0, imgSize.width).rowRange(0, imgSize.height));
	img[1].copyTo(canvas1.colRange(imgSize.width, 2 * imgSize.width).rowRange(0, imgSize.height));

	//drawMatchingPoints2(canvas1, imagepoint1, imagepoint2);
	drawMatchingPoints2(canvas1, vImagePoint1, vImagePoint2);
	 
	Mat mDepth;
	//disparity2depth(disparity, mDepth, )
	
	imshow("original image", canvas1);
	imshow("rectification", canvas);
	imshow("disparity", disparity8);
	waitKey(0);

}

void disparity2depth(Mat& mDisparity, Mat& mDepth, float fFocal, Mat& tvec1, Mat& tvec2){

}


void detectCorrespondence(Mat& img1, Mat& img2, vector<Point2f>& corner1, vector<Point2f>& corner2, int nPtMax){
	/// Parameters for Shi-Tomasi algorithm
	vector<Point2f> prevPts, nextPts;
	double qualityLevel = 0.01;
	double minDistance = 20;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	int maxCorners = nPtMax;

	/// Apply corner detection
	goodFeaturesToTrack(img1, prevPts, maxCorners, qualityLevel, minDistance,
		Mat(), blockSize, useHarrisDetector, k);

	vector<uchar> status;
	vector<float> err;

	Size winSize(51, 51);

	

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

	calcOpticalFlowPyrLK(img1, img2, prevPts, nextPts, status, err, winSize, 3, termcrit, 0, 0.001);

	corner1.clear();
	corner2.clear();
	for (int i = 0; i < prevPts.size();i++)
		if (status[i]){
		corner1.push_back(prevPts[i]);
		corner2.push_back(nextPts[i]);
		}
}