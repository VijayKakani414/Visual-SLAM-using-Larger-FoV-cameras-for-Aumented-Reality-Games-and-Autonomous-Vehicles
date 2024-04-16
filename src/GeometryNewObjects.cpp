#pragma once

#include < stdio.h>  
#include < opencv2\opencv.hpp>

#include "time.h"
#include < vector> 


#define PI 3.14159265359

using namespace std;
using namespace cv;

//structure for tracking points
struct TrackPoint
{
	Point2f	imagePoint;
	Point2f imagePointNext;	// For estimation
	Point2f projectedPoint;	// Projected points after pose estimation
	Point3f	worldPoint;
	bool	bActive;
};


//structure for camera
struct CameraParams
{
	Mat mIntrinsicMatrix;
	Mat mDistortionCoeffs;
	Vec3f rvec, tvec;
	Mat mR;
	Mat mT;
	Mat mCamera3DPos;
};

//structure of 3D model loaded from matlab
struct AlgParams
{
	Mat mVertices;
	Mat mFaces;
	vector<float> vVerticeDepth;
	vector<Point2f> vVertices2D;
	Mat mDepthGT;
	Mat mFrame;
};

void fprintMatrix(Mat matrix, string name);
void fprintfVectorMat(vector< Mat> matrix, string name);
void fprintf2Point(vector< vector< Point2f> > Points, string name);
void fprintf3Point(vector< vector< Point3f> > Points, string name);
bool detectKeyPoints(const Mat& img, vector<TrackPoint>& trackList, bool bShow = false);
void generateWorldPoints(vector<Point3f>& carpetWorldPoints);
void generateWorldPointsTest(vector<TrackPoint>& trackList);


void ComputeAngle(Point2f p1, Point2f p2, float& ang_rad, float& ang_deg);
void FillHoles(const Mat& i_in, Mat& i_out);
void showResult(CameraParams& camParams, vector<TrackPoint>& trackList, Mat& img, int tTime, int flag);
bool CarpetDetection(const vector<Point>& contour, vector<Point2f>& imagePoints);
bool DetectMorePoints(const vector<Point>& contour, vector<Point2f>& imagePoints);
Point2f rectangularity(vector<Point>& contour);									/// Compute the rectangularity of a blob
Point2f lineIntersection(Point2f A, Point2f B, Point2f C, Point2f D);
bool ClockDetect(const Mat& img, const vector<Point2f>& contour);
Vec4f LineThru2Points(Point2f& p1, Point2f& p2);

bool RefineCorners(Mat& img, vector<Point2f>& imagePoints, int ksize);

bool opticalFlowTracker(const Mat& mPrev, const Mat& mNext, vector<TrackPoint>& trackList);

void CameraIntrinsicCali(VideoCapture& cap, int nViews);
void drawDetectedCarpet(Mat& image, vector<Point2f>& imagePoints);

void showRT(Mat& frame, Mat& mR, Mat& mT, int t);
void InitializeTracker(CameraParams& camParams, vector<TrackPoint>& tracList, Mat& img, Mat& mGray, int nSz, int nLoop);
void estimateCamPose(vector<TrackPoint>& trackList, CameraParams& camParams);

void validateTrackerList(vector<TrackPoint>& trackList, CameraParams& camParams, Mat& img, float fBorderGap);

double ED(Point2f p1, Point2f p2){
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

///////////////////////////////function for phase 2////////////////////////////////////////////////
void cvtArrMat(vector<TrackPoint>& trackList, Mat& wPoints, Mat& iPoints);

void resultOfPhase1(string videoname, CameraParams &camParams, AlgParams &algParams);

void readMetaInfo(string st, Mat& imagePoints1, Mat& imagePoints2, CameraParams &camParams1, CameraParams &camParams2, AlgParams &algParams);

void drawMatchingPoints(Mat& frame, const Mat& imagePoints1, const Mat& imagePoints2);

float distance3D(Mat& p1, Mat& p2);

float distance2D(Point2f& p1, Point2f& p2);

void drawLineThru3Points(Mat& image, Mat& points, Scalar color);

void drawVerticesnFaces(Mat& verticesnFace, vector<Point2f>& vpoints2f, Mat& faces);

bool checkPointInsideImg(Mat& frame, Point2f& point2f);

void distance3Ds(AlgParams& algParams, CameraParams& camParams);

float signPoint(Point2f p1, Point2f p2, Point2f p3);

bool pointInTriangle(Point2f p, Point2f p1, Point2f p2, Point2f p3);

void computeDepthMap(AlgParams& algParams, int step);

void computeDepthMap2(AlgParams& algParams);

float depthInterpolation(Point2f p, Point2f p1, float d1, Point2f p2, float d2, Point2f p3, float d3);

Point2f findIntersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2);

bool checkFaces(Mat& r1, Mat& r2, Mat& r);

void edgeDetection(Mat& inputImg, Mat& outputImg);

////////////////////////////function for phase 2////////////////////////////////////////////////



void main()
{
	//new object detection
	//build 2D edges image from given 3D model at the specific frame
	//edge extraction from that frame (Sobel, Canny)
	//subtract edge images to get approximate area of new object

	//new object 3D geometry inference
	//compute the disparity map using a pair of images
	//infer the 3D geometry of new object (vertices and faces)

	Mat img1, img2, g1, g2, canvas;
	CameraParams camParams1, camParams2;
	Mat imagePoints1, imagePoints2;
	Mat tmp, tmp1;
	int rows, cols;
	AlgParams algParams;
	string videoname = "20160202_164235.mp4";
	////Load, intrinsic, distortion, 3D geometry (vertices, faces BENT provided)
	readMetaInfo("phase2", imagePoints1, imagePoints2, camParams1, camParams2, algParams);
	//convert to out coordinate system, right hand side

	//algParams.mVertices = (10.0f / 3.0f)*algParams.mVertices;
	//tmp = algParams.mVertices.col(2).clone();
	//tmp = 5000 - tmp;
	//tmp.copyTo(algParams.mVertices.col(2));

	Mat mV = algParams.mVertices;

	resultOfPhase1(videoname, camParams1, algParams);

	//////////////////////////phase2//////////////////////////////
	img1 = imread("phase2//1image.jpg");
	img2 = imread("phase2//2image.jpg");
	rows = img1.rows;	cols = img1.cols;
	canvas = Mat::zeros(Size(2 * cols, rows), CV_8UC3);
	img1.copyTo(canvas.colRange(0, cols).rowRange(0, rows));
	img2.copyTo(canvas.colRange(cols, 2 * cols).rowRange(0, rows));


	drawMatchingPoints(canvas, imagePoints1, imagePoints2);


	imshow("canvas", canvas);
	waitKey(0);

	//////////////////////////phase2//////////////////////////////
}

void edgeDetection(Mat& inputImg, Mat& outputImg){
	Mat gray;

	//using sobel
	//Mat grad_x, grad_y;
	//Mat abs_grad_x, abs_grad_y;
	//GaussianBlur(inputImg, inputImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//cvtColor(inputImg, gray, CV_BGR2GRAY);	

	///// Generate grad_x and grad_y
	///// Gradient X
	////Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//Sobel(inputImg, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);

	///// Gradient Y
	////Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	//Sobel(inputImg, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	//convertScaleAbs(grad_y, abs_grad_y);

	///// Total Gradient (approximate)
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImg);
	//cvtColor(outputImg, outputImg, CV_RGB2GRAY);
	//threshold(outputImg, outputImg, 20, 255, CV_THRESH_BINARY);

	//using canny
	cvtColor(inputImg, gray, CV_RGB2GRAY);
	Canny(gray, outputImg, 50, 150);

}

void resultOfPhase1(string videoname, CameraParams &camParams, AlgParams &algParams){

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//string huanPath = "E:\\2016\\BENT\\Implementation\\Data\\Videos\\S5\\";
	vector<TrackPoint> trackList;
	Mat frameTmp, verticesnFace;
	int rows, cols;
	//vector<Point2f> vPoint2f;



	VideoCapture cap(videoname);
	if (!cap.isOpened())  // check if we succeeded
		return;

	cols = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	rows = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	verticesnFace = Mat::zeros(Size(cols, rows), CV_8UC1);
	/*imshow("depth map", depthMap);
	waitKey(0);*/


	//cap >> frame; // get a new frame from camera
	//imshow("Pattern", frame);
	//cvWaitKey(0);
	Mat edgeImg, edgeDepthImg;
	bool bFirst = true;
	Mat mGrayPrev, mGrayNext;

	generateWorldPointsTest(trackList);

	int nCnt = 0;
	char c;
	int idx = 0, step = 1;
	double m, M;
	
	//vertices of our scene
	/*float test[21] = { 0, 0, 0, 1500, 0, 0, 0, 2000, 0, 0, 0, 4000, 0, 2000, 4000, 1500, 2000, 0, 1500, 0, 4000};*/
	//float test[45] = { 0, 0, 0, 1500, 0, 0, 0, 2000, 0, 0, 0, 4000, 0, 2000, 4000, 1500, 2000, 0, 1500, 0, 4000, 1975, 500, 2790, 2525, 500, 2790, 2525, 500, 2250, 1975, 500, 2250, 1975, 450, 2790, 2525, 400, 2790, 2525, 400, 2250, 1975, 400, 2250 };


	/*float test[165] = { 0, 0, 0, 1500, 0, 0, 0, 2000, 0, 0, 0, 4000, 0, 2000, 4000, 1500, 2000, 0, 1500, 0, 4000, 2025, 0, 2790, 2025, 0, 2740, 1975, 0, 2740, 1975, 0, 2790, 2525, 0, 2790, 2525, 0, 2740, 2475, 0, 2740, 2475, 0, 2790, 2525, 0, 2300, 2525, 0, 2250, 2475, 0, 2250, 2475, 0, 2300, 2025, 0, 2300, 2025, 0, 2250, 1975, 0, 2250, 1975, 0, 2300, 2025, 500, 2790, 2025, 500, 2740, 1975, 500, 2740, 1975, 500, 2790, 2525, 500, 2790, 2525, 500, 2740, 2475, 500, 2740, 2475, 500, 2790, 2525, 500, 2300, 2525, 500, 2250, 2475, 500, 2250, 2475, 500, 2300, 2025, 500, 2300, 2025, 500, 2250, 1975, 500, 2250, 1975, 500, 2300, 2025, 450, 2790, 2025, 450, 2740, 1975, 450, 2740, 1975, 450, 2790, 2525, 450, 2790, 2525, 450, 2740, 2475, 450, 2740, 2475, 450, 2790, 2525, 450, 2300, 2525, 450, 2250, 2475, 450, 2250, 2475, 450, 2300, 2025, 450, 2300, 2025, 450, 2250, 1975, 450, 2250, 1975, 450, 2300 };*/
	//float test1[18] = { 1, 2, 3, 2, 3, 6, 1, 2, 4, 2, 4, 7, 1, 3, 4, 3, 4, 5};
	//float test1[48] = { 1, 2, 3, 2, 3, 6, 1, 2, 4, 2, 4, 7, 1, 3, 4, 3, 4, 5, 8, 9, 10, 8, 10, 11, 8, 12, 9, 12, 9, 13, 9, 13, 10, 13, 14, 10, 10, 11, 14, 11, 14, 15, 15, 11, 8, 15, 12, 8 };
	//Mat vtest(15, 3, CV_32FC1, test);
	//Mat vtest1(16, 3, CV_32FC1, test1);
	//vtest.copyTo(algParams.mVertices);
	//vtest1.copyTo(algParams.mFaces);

	//Mat imageUndistorted;
	algParams.mDepthGT.create(rows, cols, CV_32F);

	//Mat& mDepth = algParams.mDepthGT;

	for (;;)
	{
		
		//grab the first frame
		cap >> algParams.mFrame; // get a new frame from camera
		frameTmp = algParams.mFrame.clone();
		verticesnFace.setTo(0);
		if (!algParams.mFrame.data)
			return;

		int t0 = clock();
		int t = 0;

		cvtColor(algParams.mFrame, mGrayNext, CV_BGR2GRAY);

		//undistort(frame, imageUndistorted, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs);

		if (bFirst){	// Initialize the trackers
			bFirst = false;
			detectKeyPoints(algParams.mFrame, trackList);
			//RefineCorners(mGrayNext, trackList, 10);
			estimateCamPose(trackList, camParams);
			//InitializeTracker(camParams, trackList, imageUndistorted, mGrayNext, 1);
			InitializeTracker(camParams, trackList, algParams.mFrame, mGrayNext, 9, 1);
			// Update the pose with more points
			estimateCamPose(trackList, camParams);
			//InitializeTracker(camParams, trackList, imageUndistorted, mGrayNext, 2);
			InitializeTracker(camParams, trackList, algParams.mFrame, mGrayNext, 11, 2);
			// Update the pose with more points
			estimateCamPose(trackList, camParams);


			/////////////////////////////phase 2/////////////////////////////////////


			//project3Dto2D(depthMap, algParams, camParams, vPoint2f);

			projectPoints(algParams.mVertices, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, algParams.vVertices2D);

			drawVerticesnFaces(algParams.mFrame, algParams.vVertices2D, algParams.mFaces);


			distance3Ds(algParams, camParams);
			//computeDepthMap(algParams, step);
			computeDepthMap2(algParams);

			minMaxLoc(algParams.mDepthGT, &m, &M);
			algParams.mDepthGT = (M - algParams.mDepthGT) / (M - m);

			//compute the edge information of frame and GT depth map
			
			edgeDetection(frameTmp, edgeImg);
			edgeDepthImg = algParams.mDepthGT * 255;
			edgeDepthImg.convertTo(edgeDepthImg, CV_8UC1);			
			Canny(edgeDepthImg, edgeDepthImg, 50, 150);
			imshow("edge", edgeImg);
			imshow("edge of depth", edgeDepthImg);

			waitKey(0);


			imshow("Depth GT", algParams.mDepthGT);
			waitKey(1);

			/////////////////////////////phase 2/////////////////////////////////////




			t = clock() - t0;
			mGrayPrev = mGrayNext.clone();
			//showResult(camParams, trackList, imageUndistorted, t, 0);
			showResult(camParams, trackList, algParams.mFrame, t, 0);
			imshow("Pattern", algParams.mFrame);
			c = char(waitKey(1));


		}
		else
		{
			// Find back the points by optical flow
			opticalFlowTracker(mGrayPrev, mGrayNext, trackList);
			printf("Frame %d\n", nCnt);
			estimateCamPose(trackList, camParams);



			////////////////////////////phase 2/////////////////////////////////////			
			//project3Dto2D(depthMap, algParams, camParams, vPoint2f);
			//project3Dto2D(depthMap, vtest, camParams, vPoint2f);
			projectPoints(algParams.mVertices, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, algParams.vVertices2D);

			drawVerticesnFaces(algParams.mFrame, algParams.vVertices2D, algParams.mFaces);
			//drawVerticesnFaces(verticesnFace, algParams.vVertices2D, algParams.mFaces);

			distance3Ds(algParams, camParams);
			computeDepthMap2(algParams);
			//computeDepthMap(algParams, step);
			minMaxLoc(algParams.mDepthGT, &m, &M);
			algParams.mDepthGT = (M - algParams.mDepthGT) / (M - m);

			//compute the edge information of frame and GT depth map

			edgeDetection(frameTmp, edgeImg);
			Canny(edgeDepthImg, edgeDepthImg, 50, 150);
			imshow("edge", edgeImg);
			imshow("edge of depth", edgeDepthImg);

			waitKey(0);

			//save the metadata of the wanted frames
			if (c == 's' || c == 'S'){
				idx++;
				string st;
				//save the image
				st = format("%dimage.jpg", idx);
				imwrite(st, frameTmp);
				cout << "the image was saved successfully" << endl;
				cv::FileStorage fs;
				//save the rotation vector
				st = format("%drotation.yml", idx);
				fs.open(st, cv::FileStorage::WRITE);
				fs << "rotation vector" << camParams.mR;
				//fs.release();
				//save the translation vector
				st = format("%dtranslation.yml", idx);
				//cv::FileStorage fs;
				fs.open(st, cv::FileStorage::WRITE);
				fs << "translation vector" << camParams.mT;
				//fs.release();				
				Mat wPoints, iPoints;
				cvtArrMat(trackList, wPoints, iPoints);
				//save world points
				st = format("%dworldPoints.yml", idx);
				//cv::FileStorage fs;
				fs.open(st, cv::FileStorage::WRITE);
				fs << "world points " << wPoints;
				//fs.release();
				//save image points
				st = format("%dimagePoints.yml", idx);
				//cv::FileStorage fs;
				fs.open(st, cv::FileStorage::WRITE);
				fs << "image points " << iPoints;
				//fs.release();
			}
			/////////////////////////////phase 2/////////////////////////////////////


			int t = clock() - t0;
			validateTrackerList(trackList, camParams, mGrayNext, 15);
			for (int i = 0; i < (int)trackList.size(); i++){
				trackList[i].imagePoint = trackList[i].imagePointNext;
			}


			mGrayNext.copyTo(mGrayPrev);
			showResult(camParams, trackList, algParams.mFrame, t, 1);
			imshow("Pattern", algParams.mFrame);
			c = char(cvWaitKey(1));
			nCnt++;


		}

		imshow("Depth GT", algParams.mDepthGT);
		imshow("Faces GT", verticesnFace);
		waitKey(1);

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////  
}


//////////////////////////////////////////////////=================================//////////////////////
float distance2D(Point2f& p1, Point2f& p2){
	float d = 0, tmp;
	tmp = p1.x - p2.x;
	d = d + tmp*tmp;
	tmp = p1.y - p2.y;
	d = d + tmp*tmp;
	return sqrt(d);
}


Point2f findIntersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2){
	Point2f r;
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;

	return r;
}

float depthInterpolation(Point2f p, Point2f p1, float d1, Point2f p2, float d2, Point2f p3, float d3){
	float d, td;
	Point2f tpoint;
	float k1, k2, k3;

	tpoint = findIntersection(p, p1, p2, p3);
	k2 = distance2D(tpoint, p2);
	k3 = distance2D(tpoint, p3);
	d = (k2*d3 + k3*d2) / (k2 + k3);
	k1 = distance2D(p, p1);
	k2 = distance2D(p, tpoint);
	d = (k1*d + k2*d1) / (k1 + k2);

	return d;
}

void computeDepthMap2(AlgParams& algParams){
	int rows, cols;
	Point2f p;

	vector<float>& vDistances = algParams.vVerticeDepth;
	vector<Point2f>& vPoints2f = algParams.vVertices2D;
	Mat& depthMap = algParams.mDepthGT;

	Mat mMask;
	mMask.create(depthMap.size(), CV_8U);


	depthMap.create(algParams.mFrame.size(), CV_32F);

	depthMap.setTo(FLT_MAX);

	Vec3i faceIdx;
	vector<Point2f> face2D;
	Vec3f faceDepth;

	face2D.resize(3);

	rows = depthMap.rows;	cols = depthMap.cols;

	float fVal;
	float* p32fFace = (float*)algParams.mFaces.data;

	float fXMax, fYMax, fXMin, fYMin;
	for (int k = 0; k < algParams.mFaces.rows; k++){
		for (int i = 0; i < 3; i++){
			faceIdx[i] = (int)(*p32fFace++) - 1;
			face2D[i] = algParams.vVertices2D[faceIdx[i]];
			faceDepth[i] = algParams.vVerticeDepth[faceIdx[i]];
		}
		fXMax = max(face2D[0].x, max(face2D[1].x, face2D[2].x));
		fYMax = max(face2D[0].y, max(face2D[1].y, face2D[2].y));
		fXMin = min(face2D[0].x, min(face2D[1].x, face2D[2].x));
		fYMin = min(face2D[0].y, min(face2D[1].y, face2D[2].y));

		for (float x = fXMin; x <= fXMax; x++)
			for (float y = fYMin; y <= fYMax; y++){

			if ((x < 0) || (y < 0) || (x >= cols) || (y >= rows))
				continue;
			p = Point2f(x, y);

			if (pointInTriangle(p, face2D[0], face2D[1], face2D[2])){

				Point2f tpoint = findIntersection(face2D[0], p, face2D[1], face2D[2]);
				fVal = depthInterpolation(p, face2D[0], faceDepth[0], face2D[1], faceDepth[1], face2D[2], faceDepth[2]);
				depthMap.at<float>(y, x) = min(fVal, depthMap.at<float>(y, x));
			}
			}
	}
	Mat mGap;
	mGap = (depthMap == FLT_MAX);

	Mat mKernel = getStructuringElement(CV_SHAPE_ELLIPSE, Size(3, 3));
	morphologyEx(mGap, mMask, MORPH_OPEN, mKernel);

	mGap = mGap - mMask;

	depthMap.setTo(NAN, mMask);

	Rect recWin;
	recWin.width = recWin.height = 3;
	bitwise_not(mGap, mGap);
	for (int y = 1; y < rows - 1; y++)
		for (int x = 1; x < cols - 1; x++)
			if (mGap.at<uchar>(y, x) == 0){
		recWin.x = x - 1;
		recWin.y = y - 1;
		Mat Win = depthMap(recWin).clone();
		float vmean = (float)mean(Win, mGap(recWin)).val[0];
		depthMap.at<float>(y, x) = vmean;
			}

	depthMap.row(0).setTo(NAN);
	depthMap.row(rows - 1).setTo(NAN);
	depthMap.col(0).setTo(NAN);
	depthMap.col(cols - 1).setTo(NAN);

	//double vmin, vmax;
	//minMaxLoc(depthMap, &vmin, &vmax);
	//depthMap = (vmax - depthMap) / (vmax - vmin);

	//imshow("Depth", depthMap);
	//cvWaitKey(1);
}

void computeDepthMap(AlgParams& algParams, int step){
	int rows, cols;
	Point2f p;

	vector<float>& vDistances = algParams.vVerticeDepth;
	vector<Point2f>& vPoints2f = algParams.vVertices2D;
	//Mat& depthMap = algParams.mDepthGT;
	Mat depthMap;

	depthMap.create(algParams.mFrame.size(), CV_32F);

	depthMap.setTo(FLT_MAX);

	int idx1, idx2, idx3;
	rows = depthMap.rows;	cols = depthMap.cols;
	float w1, w2, w3, tsum = 0;
	float fVal;
	for (int i = 0; i < rows; i += step){
		float* data = depthMap.ptr<float>(i);

		for (int j = 0; j < cols; j += step){
			p = Point2f(j, i);
			for (int k = 0; k < algParams.mFaces.rows; k++){
				idx1 = (int)algParams.mFaces.at<float>(k, 0) - 1;
				idx2 = (int)algParams.mFaces.at<float>(k, 1) - 1;
				idx3 = (int)algParams.mFaces.at<float>(k, 2) - 1;

				if (pointInTriangle(p, vPoints2f[idx1], vPoints2f[idx2], vPoints2f[idx3])){


					Point2f tpoint = findIntersection(vPoints2f[idx1], p, vPoints2f[idx2], vPoints2f[idx3]);
					//float d;
					fVal = depthInterpolation(p, vPoints2f[idx1], vDistances[idx1], vPoints2f[idx2], vDistances[idx2], vPoints2f[idx3], vDistances[idx3]);
					data[j] = min(fVal, data[j]);

					//break;
				}
				//depthMap.at<float>(i, j) = NAN;

			}

		}

	}
	//check if the point is inside particular face


}

float signPoint(Point2f p1, Point2f p2, Point2f p3){
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool pointInTriangle(Point2f p, Point2f p1, Point2f p2, Point2f p3){
	bool b1, b2, b3;


	b1 = signPoint(p, p1, p2) < 0.0f;
	b2 = signPoint(p, p2, p3) < 0.0f;
	b3 = signPoint(p, p3, p1) < 0.0f;

	return ((b1 == b2) && (b2 == b3));
}

void distance3Ds(AlgParams& algParams, CameraParams& camParams){

	Mat camPos;
	algParams.vVerticeDepth.resize(algParams.mVertices.rows);
	Mat mTemp;
	camPos = camParams.mCamera3DPos.clone().t();
	camPos.convertTo(camPos, CV_32F);
	for (int i = 0; i < algParams.mVertices.rows; i++){
		mTemp = camPos - algParams.mVertices.row(i);
		mTemp = mTemp.mul(mTemp);
		algParams.vVerticeDepth[i] = sqrt(sum(mTemp).val[0]);
	}
}

bool checkPointInsideImg(Mat& frame, Point2f& point2f){
	int rows, cols;
	//bool ck = false;
	rows = frame.rows; cols = frame.cols;
	if (point2f.x <0 || point2f.x > cols || point2f.y < 0 || point2f.y > rows)
		return false;
	else
		return true;
}

bool checkFaces(Mat& r1, Mat& r2, Mat& r){
	bool ck = false;
	int idx = 0;
	Mat rt1, rt2, tmp, tmp1;


	cv::sort(r1, rt1, CV_SORT_ASCENDING);
	cv::sort(r2, rt2, CV_SORT_ASCENDING);
	tmp = abs(rt2 - rt1);

	/*reduce(tmp, tmp1, 1, CV_REDUCE_SUM);
	tmp = tmp / tmp1.at<float>(0,0) ;*/
	tmp.convertTo(tmp, CV_8UC1);
	findNonZero(tmp, tmp1);

	if (tmp1.rows == 1 && tmp1.cols == 1){
		ck = true;
		int idx1 = tmp1.at<int>(0, 0);
		int idx2 = tmp1.at<int>(0, 1);

		r = Mat(1, 4, CV_32FC1);
		rt1.copyTo(r.colRange(0, 3));
		r.at<float>(0, 3) = rt2.at<float>(idx2, idx1);
		cv::sort(r, r, CV_SORT_ASCENDING);


	}


	return ck;
}

void drawVerticesnFaces(Mat& verticesnFace, vector<Point2f>& vpoints2f, Mat& faces){
	Scalar color(255, 255, 0);
	int rows, cols;
	Mat recMat;


	for (int i = 0; i < faces.rows; i++){
		int idx1, idx2, idx3, idx4;
		idx1 = (int)faces.at<float>(i, 0) - 1;
		idx2 = (int)faces.at<float>(i, 1) - 1;
		idx3 = (int)faces.at<float>(i, 2) - 1;


		if (!checkPointInsideImg(verticesnFace, vpoints2f[idx1])){
			if (!checkPointInsideImg(verticesnFace, vpoints2f[idx2])){
				if (!checkPointInsideImg(verticesnFace, vpoints2f[idx3])){
					cout << "no line to draw" << endl;
				}
				else{
					cout << "one point is inside" << endl;
				}
			}
			else{
				line(verticesnFace, vpoints2f[idx3], vpoints2f[idx2], color);
			}
		}
		else{
			line(verticesnFace, vpoints2f[idx1], vpoints2f[idx2], color);
			line(verticesnFace, vpoints2f[idx1], vpoints2f[idx3], color);
			line(verticesnFace, vpoints2f[idx2], vpoints2f[idx3], color);
		}

	}

}

void drawLineThru3Points(Mat& image, Mat& points, Scalar color){
	Point2f p1, p2;
	for (int i = 0; i < points.rows; i++){
		if (i < points.rows - 1){
			p1 = Point2f(points.at<float>(i, 0), points.at<float>(i, 1));
			p2 = Point2f(points.at<float>(i + 1, 0), points.at<float>(i + 1, 1));
		}
		else {
			p1 = Point2f(points.at<float>(i, 0), points.at<float>(i, 1));
			p2 = Point2f(points.at<float>(0, 0), points.at<float>(0, 1));
		}
		line(image, p1, p2, color);
	}
}

float distance3D(Mat& p1, Mat& p2){
	if (p1.rows != p2.rows || p1.cols != p2.cols)
		return -1;
	float d = 0, temp;

	temp = ((float)p1.at<double>(0, 0) - p2.at<float>(0, 0));
	temp = temp*temp;
	d = d + temp;
	temp = ((float)p1.at<double>(0, 1) - p2.at<float>(0, 1));
	temp = temp*temp;
	d = d + temp;
	temp = ((float)p1.at<double>(0, 2) - p2.at<float>(0, 2));
	temp = temp*temp;
	d = d + temp;

	return sqrt(d);
}



void drawMatchingPoints(Mat& canvas, const Mat& imagePoints1, const Mat& imagePoints2){
	//draw lines and circles
	int rows, cols;
	RNG rng(0xFFFFFFFF);
	rows = canvas.rows;	cols = canvas.cols / 2;
	for (int i = 0; i < imagePoints1.rows; i++){
		Point2f p1, p2;
		p1 = Point2f((float)imagePoints1.at<double>(i, 0), (float)imagePoints1.at<double>(i, 1));
		p2 = Point2f((float)imagePoints2.at<double>(i, 0) + cols, (float)imagePoints2.at<double>(i, 1));
		circle(canvas, p1, 2, Scalar(0, 0, 255));
		circle(canvas, p2, 2, Scalar(0, 0, 255));
		line(canvas, p1, p2, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));

	}
}

void readMetaInfo(string st, Mat& imagePoints1, Mat& imagePoints2, CameraParams &camParams1, CameraParams &camParams2, AlgParams &algParams){

	string tmp = "";
	cv::FileStorage fs;
	//read the intrinsic and distortion
	tmp = st + "//IntrinsicMatrix.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["Intrinsic matrix"] >> camParams1.mIntrinsicMatrix;
	fs["Intrinsic matrix"] >> camParams2.mIntrinsicMatrix;
	tmp = "";
	tmp = st + "//DistortionCoeffs.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["Distortion coeffs matrix"] >> camParams1.mDistortionCoeffs;
	fs["Distortion coeffs matrix"] >> camParams2.mDistortionCoeffs;

	//read the rotation vector
	tmp = "";
	tmp = st + "//1rotation.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["rotation vector"] >> camParams1.mR;
	tmp = "";
	tmp = st + "//2rotation.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["rotation vector"] >> camParams2.mR;


	//read the translation vector
	tmp = "";
	tmp = st + "//1translation.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["translation vector"] >> camParams1.mT;
	tmp = "";
	tmp = st + "//2translation.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["translation vector"] >> camParams2.mT;


	//read image points	
	tmp = "";
	tmp = st + "//1imagePoints.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["image points"] >> imagePoints1;
	tmp = "";
	tmp = st + "//2imagePoints.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["image points"] >> imagePoints2;

	tmp = "";
	tmp = st + "//Geometry.yml";
	fs = FileStorage(tmp, cv::FileStorage::READ);
	fs["Vertices"] >> algParams.mVertices;
	fs["Faces"] >> algParams.mFaces;
	fs.release();



}


void cvtArrMat(vector<TrackPoint>& trackList, Mat& wPoints, Mat& iPoints){
	int num = 0;
	for (size_t i = 0; i < trackList.size(); i++){
		if (trackList[i].bActive)
			num++;
	}

	wPoints = Mat(num, 3, CV_64FC1);
	iPoints = Mat(num, 2, CV_64FC1);

	int idx = 0;
	for (size_t i = 0; i < trackList.size(); i++){
		if (trackList[i].bActive){
			wPoints.at<double>(idx, 0) = trackList[i].worldPoint.x;
			wPoints.at<double>(idx, 1) = trackList[i].worldPoint.y;
			wPoints.at<double>(idx, 2) = trackList[i].worldPoint.z;

			iPoints.at<double>(idx, 0) = trackList[i].imagePoint.x;
			iPoints.at<double>(idx, 1) = trackList[i].imagePoint.y;
			idx++;
		}
	}
}


void validateTrackerList(vector<TrackPoint>& trackList, CameraParams& camParams, Mat& img, float fBorderGap){
	vector<Point2f> imagePts, imagePts1;
	vector<Point3f> worldPts, worldPts1;

	for (int i = 0; i < (int)trackList.size(); i++){
		imagePts.push_back(trackList[i].imagePoint);
		worldPts.push_back(trackList[i].worldPoint);
		/*if (trackList[i].bActive){
		imagePts1.push_back(trackList[i].imagePoint);
		worldPts1.push_back(trackList[i].worldPoint);
		}*/
	}

	projectPoints(worldPts, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, imagePts);

	for (int i = 0; i < (int)trackList.size(); i++){

		trackList[i].projectedPoint = imagePts[i];
		if (trackList[i].bActive){

			// Check if the active point is near the border, then make it inactive

			if ((imagePts[i].x < fBorderGap) || (imagePts[i].x + fBorderGap > img.cols) || (imagePts[i].y < fBorderGap) || (imagePts[i].y + fBorderGap > img.rows))
				trackList[i].bActive = false;
		}

		// If the distance between the projected to the tracking point is too far, then make it inactive
		if (ED(trackList[i].imagePointNext, trackList[i].projectedPoint) > 15)
			trackList[i].bActive = false;
	}

	// Activate inactive points to the track-list
	vector<Point2f> vImagePts;
	vector<int> vIdx;
	fBorderGap *= 2;
	for (int i = 0; i < (int)trackList.size(); i++)
		if (!trackList[i].bActive)
			if ((trackList[i].projectedPoint.x > fBorderGap) && (trackList[i].projectedPoint.x + fBorderGap < img.cols))
				if ((trackList[i].projectedPoint.y > fBorderGap) && (trackList[i].projectedPoint.y + fBorderGap < img.rows)){
		vImagePts.push_back(trackList[i].projectedPoint);
		vIdx.push_back(i);
				}
	if (vImagePts.size() == 0)
		return;

	RefineCorners(img, vImagePts, 10);
	for (int i = 0; i < (int)vImagePts.size(); i++){
		trackList[vIdx[i]].bActive = true;
		trackList[vIdx[i]].imagePointNext = vImagePts[i];
	}

};

void estimateCamPose(vector<TrackPoint>& trackList, CameraParams& camParams){
	vector<Point2f> imagePts;
	vector<Point3f> worldPts;
	Mat R;
	Rodrigues(camParams.mR, R);
	camParams.mCamera3DPos = -1 * R.t()*camParams.mT;

	for (int i = 0; i < (int)trackList.size(); i++)
		if (trackList[i].bActive){
		imagePts.push_back(trackList[i].imagePoint);
		worldPts.push_back(trackList[i].worldPoint);
		}
	if ((int)imagePts.size() < 4)
		return;

	solvePnP(worldPts, imagePts, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, camParams.mR, camParams.mT);

}

void InitializeTracker(CameraParams& camParams, vector<TrackPoint>& tracList, Mat& img, Mat& mGray, int nSz, int nLoop){

	for (int i = 0; i < 4; i++)
		circle(img, tracList[i].imagePoint, 3, Scalar(0, 255, 255));

	// Project the origin
	vector<Point3f> vObjectPoints;
	vObjectPoints.push_back(Point3f(0, 0, 0));
	vObjectPoints.push_back(Point3f(1000, 0, 0));
	vObjectPoints.push_back(Point3f(0, 2000, 0));
	vObjectPoints.push_back(Point3f(0, 0, 4000));

	vector<Point2f> vImagePoints;
	projectPoints(vObjectPoints, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, vImagePoints);

	Scalar color(0, 255, 255);
	// Draw the axes
	circle(img, vImagePoints[0], 5, color, 5);
	circle(img, vImagePoints[1], 3, color, 3);
	circle(img, vImagePoints[2], 3, color, 3);
	circle(img, vImagePoints[3], 3, color, 3);
	line(img, vImagePoints[0], vImagePoints[1], color);
	line(img, vImagePoints[0], vImagePoints[2], color);
	line(img, vImagePoints[0], vImagePoints[3], color);

	//////////////////////////////////////////////////////////////////////////
	if (nLoop == 1){
		vObjectPoints.clear();
		vector<int> vIdx;
		for (int i = 0; i < 7; i++)
			if (!tracList[i].bActive){
			vObjectPoints.push_back(tracList[i].worldPoint);
			vIdx.push_back(i);
			}


		projectPoints(vObjectPoints, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, vImagePoints);


		for (int i = 0; i < (int)vImagePoints.size(); i++)
			circle(img, vImagePoints[i], 3, Scalar(0, 255, 255));

		// Detect object based on the corner
		RefineCorners(mGray, vImagePoints, nSz);

		for (int i = 0; i < (int)vImagePoints.size(); i++)
			circle(img, vImagePoints[i], 3, Scalar(0, 0, 255));

		for (int i = 0; i < (int)vImagePoints.size(); i++)
		{
			tracList[vIdx[i]].worldPoint = vObjectPoints[i];
			tracList[vIdx[i]].imagePoint = vImagePoints[i];
			tracList[vIdx[i]].bActive = true;
		}
	}
	else
	{
		vObjectPoints.clear();
		vector<int> vIdx;
		for (int i = 0; i < (int)tracList.size(); i++)
			if (!tracList[i].bActive){
			vObjectPoints.push_back(tracList[i].worldPoint);
			vIdx.push_back(i);
			}


		projectPoints(vObjectPoints, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, vImagePoints);


		for (int i = 0; i < (int)vImagePoints.size(); i++)
			circle(img, vImagePoints[i], 3, Scalar(0, 255, 255));

		// Detect object based on the corner
		RefineCorners(mGray, vImagePoints, nSz);

		for (int i = 0; i < (int)vImagePoints.size(); i++)
			circle(img, vImagePoints[i], 3, Scalar(0, 0, 255));

		for (int i = 0; i < (int)vImagePoints.size(); i++)
		{
			tracList[vIdx[i]].worldPoint = vObjectPoints[i];
			tracList[vIdx[i]].imagePoint = vImagePoints[i];
			tracList[vIdx[i]].bActive = true;
		}
	}
}

bool opticalFlowTracker(const Mat& mPrev, const Mat& mNext, vector<TrackPoint>& trackList){

	Mat frame, grayFrames, rgbFrames, prevGrayFrame;

	Point2f diff;

	vector<uchar> status;
	vector<float> err;

	Size winSize(31, 31);

	vector<Point2f> prevPts, nextPts;
	for (int i = 0; i < (int)trackList.size(); i++)
		if (trackList[i].bActive)
			prevPts.push_back(trackList[i].imagePoint);

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	if ((int)prevPts.size() == 0)
		return false;
	calcOpticalFlowPyrLK(mPrev, mNext, prevPts, nextPts,
		status, err, winSize, 3, termcrit, 0, 0.001);

	int nIdx = 0;
	for (int i = 0; i < (int)trackList.size(); i++)
		if (trackList[i].bActive){
		trackList[i].imagePointNext = nextPts[nIdx++];
		}

	return true;
};

//void CameraIntrinsicCali(VideoCapture& cap, int nViews){
//	vector< vector< Point2f> > imagePointsSet;
//	vector< Point2f> imagePoints;
//	vector< vector< Point3f> > objectPointsSet;
//	vector< Point3f> carpetWorldPoints;
//
//	//Set input params..  
//	int n_boards;
//
//
//	n_boards = nViews;
//	//////////////////////////////////////////////////////////////////////////////////////////////////  
//	//image load  
//	//extraction image point and object point  
//	//////////////////////////////////////////////////////////////////////////////////////////////////  
//	generateWorldPoints(carpetWorldPoints);
//
//	int nCnt = 0;
//
//	Mat frame;
//	//VideoCapture cap("E:\\2016\\BENT\\Implementation\\Data\\Videos\\Note3\\20160111_150957.mp4"); 
//	//VideoCapture cap("E:\\2016\\BENT\\Implementation\\Data\\Videos\\Note3\\20160111_150902.mp4"); // open the default camera
//
//	while (true)
//	{
//
//		printf("%d\n", nCnt);
//
//		//img = imread(str, 1);
//		cap >> frame;
//		if (frame.rows == 0)
//			break;
//
//		//imshow("Input", img);
//		//cvWaitKey(0);
//
//		detectKeyPoints(frame, imagePoints, true);
//
//			
//		for (int i = 0; i < (int)imagePoints.size(); i++)
//			circle(frame, imagePoints[i], 3, Scalar(0, 0, 255));
//
//		imshow("Pattern", frame);
//		cvWaitKey(500);
//
//		if ((int)imagePoints.size() > 0){
//			imagePointsSet.push_back(imagePoints);
//			objectPointsSet.push_back(carpetWorldPoints);
//
//			nCnt++;
//		}
//			
//
//
//
//		if (nCnt > n_boards)
//			break;
//
//		for (int i = 0; i < 60; i++)
//			cap >> frame;
//	}
//
//	if (frame.rows == 0)
//		return;
//
//	//calibration part  
//	vector< Mat> rvecs, tvecs;
//	Mat mIntrinsicMatrix(3, 3, CV_64F);
//	Mat mDistortionCoeffs(8, 1, CV_64F);
//
//	calibrateCamera(objectPointsSet, imagePointsSet, frame.size(), mIntrinsicMatrix, mDistortionCoeffs, rvecs, tvecs);
//
//	// Save the intrinsic and distortion coeffs to file
//	cv::FileStorage fileIntrinsicMatrix;
//	fileIntrinsicMatrix.open("IntrinsicMatrix.yml", cv::FileStorage::WRITE);
//	fileIntrinsicMatrix << "Intrinsic matrix" << mIntrinsicMatrix;
//	fileIntrinsicMatrix.release();
//
//	cv::FileStorage fileDistortionCoeffs;
//	fileDistortionCoeffs.open("DistortionCoeffs.yml", cv::FileStorage::WRITE);
//	fileDistortionCoeffs << "Distortion coeffs matrix" << mDistortionCoeffs;
//	fileDistortionCoeffs.release();
//
//	//////////////////////////////////////////////////////////////////////////////////////////////////  
//	//save part  
//	fprintMatrix(mIntrinsicMatrix, "intrinsic.txt");
//	fprintMatrix(mDistortionCoeffs, "distortion_coeffs.txt");
//
//	fprintfVectorMat(rvecs, "rotation.txt");
//	fprintfVectorMat(tvecs, "translation.txt");
//
//};

bool RefineCorners(Mat& img, vector<Point2f>& imagePoints, int ksize){
	// b4Active = true: refine on active points; otherwise: in-active points

	/// Parameters for Shi-Tomasi algorithm
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	int maxCorners = 1;

	Rect win;
	win.width = 2 * ksize;
	win.height = 2 * ksize;
	Mat mWin;
	for (int i = 0; i < (int)imagePoints.size(); i++){
		win.x = (int)(imagePoints[i].x - ksize);
		win.y = (int)(imagePoints[i].y - ksize);
		win.x = max(0, win.x); win.x = min(img.cols - 2 * ksize, win.x);
		win.y = max(0, win.y); win.y = min(img.rows - 2 * ksize, win.y);

		mWin = img(win).clone();

		/// Apply corner detection
		goodFeaturesToTrack(mWin, corners, maxCorners, qualityLevel, minDistance,
			Mat(), blockSize, useHarrisDetector, k);
		if ((int)corners.size()>0)
			imagePoints[i] = corners[0] + imagePoints[i] - Point2f((float)ksize, (float)ksize);
	}

	return true;
}

Point2f lineIntersection(Point2f A, Point2f B, Point2f C, Point2f D){

	float  distAB, theCos, theSin, newX, ABpos;

	//  Fail if either line is undefined.
	if (A.x == B.x && A.y == B.y || C.x == D.x && C.y == D.y)
		return Point2f(-1, -1);

	//  (1) Translate the system so that point A is on the origin.
	B.x -= A.x; B.y -= A.y;
	C.x -= A.x; C.y -= A.y;
	D.x -= A.x; D.y -= A.y;

	//  Discover the length of segment A-B.
	distAB = sqrt(B.x*B.x + B.y*B.y);

	//  (2) Rotate the system so that point B is on the positive X axis.
	theCos = B.x / distAB;
	theSin = B.y / distAB;
	newX = C.x*theCos + C.y*theSin;
	C.y = C.y*theCos - C.x*theSin; C.x = newX;
	newX = D.x*theCos + D.y*theSin;
	D.y = D.y*theCos - D.x*theSin; D.x = newX;

	//  Fail if the lines are parallel.
	if (C.y == D.y) return Point2f(-1, -1);

	//  (3) Discover the position of the intersection point along line A-B.
	ABpos = D.x + (C.x - D.x)*D.y / (D.y - C.y);

	//  (4) Apply the discovered position to line A-B in the original coordinate system.
	Point2f ptCross;
	ptCross.x = A.x + ABpos*theCos;
	ptCross.y = A.y + ABpos*theSin;

	//  Success.
	return ptCross;
}

bool ClockDetect(const Mat& img, const vector<Point2f>& carpetCorners){
	vector<Point2f> contourROI;
	return true;
};


void generateWorldPoints(vector<Point3f>& carpetWorldPoints){
	carpetWorldPoints.clear();

	carpetWorldPoints.push_back(Point3f(1000, 1645, 0));
	carpetWorldPoints.push_back(Point3f(1970, 1645, 0));
	carpetWorldPoints.push_back(Point3f(1965, 990, 0));
	carpetWorldPoints.push_back(Point3f(990, 1000, 0));
	if (carpetWorldPoints.size() == 7){
		carpetWorldPoints.push_back(Point3f(1000, 0, 0));
		carpetWorldPoints.push_back(Point3f(1970, 0, 0));
		carpetWorldPoints.push_back(Point3f(2473, 0, 0));
	}
};

void generateWorldPointsTest(vector<TrackPoint>& trackList){
	trackList.clear();
	TrackPoint aPoint;
	// The carpet corners
	aPoint.worldPoint = (Point3f(0, 1645, 4000));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(0, 1645, 3030));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(0, 990, 3035));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(0, 1000, 4010));
	trackList.push_back(aPoint);

	// The clock center
	aPoint.worldPoint = (Point3f(0, 1505, 1503));
	trackList.push_back(aPoint);

	// The wood chair
	aPoint.worldPoint = (Point3f(530, 0, 3000));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(1000, 0, 3000));
	trackList.push_back(aPoint);

	// The high table (left view)
	aPoint.worldPoint = (Point3f(2000, 0, 3000));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(1500, 0, 3000));
	trackList.push_back(aPoint);
	//vObjectPoints.push_back(Point3f(2000, 700, 3005));
	//vObjectPoints.push_back(Point3f(1495, 700, 3005));


	//// The small white table
	aPoint.worldPoint = (Point3f(2005, 0, 2750));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(2550, 0, 2750));
	trackList.push_back(aPoint);
	//aPoint.worldPoint = (Point3f(2550, 450, 2750));
	//trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(2550, 0, 2200));
	trackList.push_back(aPoint);

	// The high table front view
	aPoint.worldPoint = (Point3f(2000, 0, 1900));
	trackList.push_back(aPoint);

	// The white chair
	aPoint.worldPoint = (Point3f(2000, 0, 1470));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(2000, 0, 1000));
	trackList.push_back(aPoint);



	for (int i = 0; i < (int)trackList.size(); i++)
		trackList[i].bActive = false;
}

Vec4f LineThru2Points(Point2f& p1, Point2f& p2){
	Vec4f aLine;
	aLine[2] = p1.x;
	aLine[3] = p1.y;
	aLine[1] = 1;
	if (p2.y == p1.y)
		aLine[0] = 0;
	else
		aLine[0] = (p2.x - p1.x) / (p2.y - p1.y);
	return aLine;
}

bool detectKeyPoints(const Mat& img, vector<TrackPoint>& trackList, bool bShow){
	//imagePoints.clear();
	Mat mDis = img;


	Mat m8U, mGrey;
	cvtColor(img, mGrey, CV_BGR2GRAY);

	double dMean = mean(mGrey).val[0];
	double dMin, dMax;
	minMaxLoc(mGrey, &dMin, &dMax);

	double dThresh = (dMin + dMean) / 2;

	m8U = mGrey < dThresh;

	//m8U(Range(0, 120), Range(0, m8U.cols)).setTo(0);
	FillHoles(m8U, m8U);


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(m8U, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	vector<Point2f>  vRectangularity;
	m8U.setTo(0);
	double dAreThreshold = img.rows*img.cols / 100;
	for (int i = 0; i < (int)contours.size(); i++)
		if (contourArea(contours[i]) > dAreThreshold){
		Point2f pt2f = rectangularity(contours[i]);
		vRectangularity.push_back(pt2f);
		//if (dval > 0.9)
		//	drawContours(m8U, contours, i, Scalar(120), CV_FILLED);
		}
		else
			vRectangularity.push_back(Point2f(0, 0));


	int nContourCnt = (int)contours.size();

	// Find the two blob that have the rectangularity close to 1
	double dRecMax = 0.;
	int nRecMaxIdx = 0, nRexSecondMax = 0;
	for (int i = 0; i < nContourCnt; i++)
		if (vRectangularity[i].y > dRecMax){
		dRecMax = vRectangularity[i].y;
		nRecMaxIdx = i;
		}

	printf("Max rectangularity: %.2f\n", vRectangularity[nRecMaxIdx]);

	if (vRectangularity[nRecMaxIdx].y < 0.8)
		return false;



	vector<Point2f> imagePoints;
	bool r = CarpetDetection(contours[nRecMaxIdx], imagePoints);

	if (r){
		RefineCorners(mGrey, imagePoints, 10);
		for (int i = 0; i < 4; i++){
			trackList[i].imagePoint = imagePoints[i];
			trackList[i].bActive = true;
		}
	}

	return r;


}

bool DetectMorePoints(const vector<Point>& contour, vector<Point2f>& imagePoints){
	vector<Point2f> corners;
	CarpetDetection(contour, corners);

	// Find cross points based on the known objects
	Point2f ptCross;
	ptCross = lineIntersection(imagePoints[0], imagePoints[3], corners[2], corners[3]);

	imagePoints.push_back(ptCross);

	ptCross = lineIntersection(imagePoints[1], imagePoints[2], corners[2], corners[3]);

	imagePoints.push_back(ptCross);

	ptCross = lineIntersection(imagePoints[0], imagePoints[2], corners[2], corners[3]);

	imagePoints.push_back(ptCross);

	//for (int i = 0; i < corners.size(); i++)
	//	imagePoints.push_back(corners[i]);

	return true;
}


Point2f rectangularity(vector<Point>& contour){
	/**
	* Parameters:
	\n	+contour: Contour of the boundary
	\n\n This function computes the rectangularity of a region given by the contour. The region is more rectangular if the score is closer to 1.
	*/
	double dCountourArea = contourArea(contour);

	//Rect rBoundingRect = boundingRect(contour);
	RotatedRect rBoundingRect = minAreaRect(contour);

	double dBoxArea = rBoundingRect.size.width*rBoundingRect.size.height;

	Point2f pt;

	pt.y = (float)(dCountourArea / dBoxArea);
	pt.x = pt.y * (float)dCountourArea;

	//pt.x = dCountourArea / dBoxArea;
	//pt.y = pt.x * dCountourArea;

	return pt;
}


bool CarpetDetection(const vector<Point>& contour, vector<Point2f>& imagePoints){

	vector<Point> contours_poly;
	approxPolyDP(Mat(contour), contours_poly, 5, true);

	if (contours_poly.size() != 4)
		return false;

	// Detect corners
	imagePoints.resize(4);
	int nIdx = 0;

	int nSumMin = INT_MAX, nSumMax = 0;
	for (int i = 0; i < 4; i++){
		if (contours_poly[i].x + contours_poly[i].y < nSumMin){
			nSumMin = contours_poly[i].x + contours_poly[i].y;
			nIdx = i;
		}
	}
	imagePoints[0] = contours_poly[nIdx];

	contours_poly.erase(contours_poly.begin() + nIdx);
	nIdx = 0;
	for (int i = 0; i < 3; i++){
		if (contours_poly[i].x + contours_poly[i].y > nSumMax){
			nSumMax = contours_poly[i].x + contours_poly[i].y;
			nIdx = i;
		}
	}
	imagePoints[2] = contours_poly[nIdx];

	contours_poly.erase(contours_poly.begin() + nIdx);

	if (contours_poly[0].x < contours_poly[1].x){
		imagePoints[3] = contours_poly[0];
		imagePoints[1] = contours_poly[1];
	}
	else
	{
		imagePoints[1] = contours_poly[0];
		imagePoints[3] = contours_poly[1];
	}
	return true;
}

void showResult(CameraParams& camParams, vector<TrackPoint>& trackList, Mat& img, int tTime, int flag){
	// Draw the axes
	vector<Point3f> vObjectPoints;
	vObjectPoints.push_back(Point3f(0, 0, 0));
	vObjectPoints.push_back(Point3f(1000, 0, 0));
	vObjectPoints.push_back(Point3f(0, 2000, 0));
	vObjectPoints.push_back(Point3f(0, 0, 4000));

	vector<Point2f> vImagePoints;

	projectPoints(vObjectPoints, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, vImagePoints);

	circle(img, vImagePoints[0], 5, Scalar(0, 0, 255), 5);
	circle(img, vImagePoints[1], 3, Scalar(0, 0, 255), 3);
	circle(img, vImagePoints[2], 3, Scalar(0, 0, 255), 3);
	circle(img, vImagePoints[3], 3, Scalar(0, 0, 255), 3);
	line(img, vImagePoints[0], vImagePoints[1], Scalar(0, 0, 255));
	line(img, vImagePoints[0], vImagePoints[2], Scalar(0, 0, 255));
	line(img, vImagePoints[0], vImagePoints[3], Scalar(0, 0, 255));

	if (flag){		// Show optical flow
		// Show the active tracking points
		for (int i = 0; i < (int)trackList.size(); i++)
			if (trackList[i].bActive){
			string st = format("%d", i);
			putText(img, st, Point((int)trackList[i].imagePoint.x + 3, (int)trackList[i].imagePoint.y + 3), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
			circle(img, trackList[i].imagePointNext, 3, Scalar(0, 0, 255));
			line(img, trackList[i].imagePoint, trackList[i].imagePointNext, Scalar(0, 0, 255));
			// Projected points
			circle(img, trackList[i].projectedPoint, 3, Scalar(0, 255, 0));
			}
	}




	string txt = format("Rotation(degree) Yaw:%.0f, Pitch:%.0f, Roll:%.0f", camParams.mR.at<double>(0, 0)*180. / PI,
		camParams.mR.at<double>(1, 0)*180. / PI, camParams.mR.at<double>(2, 0)*180. / PI);
	string txt1 = format("Translation(mm) X:%.0f, Y:%.0f, Z:%.0f", camParams.mCamera3DPos.at<double>(0, 0), camParams.mCamera3DPos.at<double>(1, 0), camParams.mCamera3DPos.at<double>(2, 0));
	//string txt1 = format("Translation: X = %.0f, Y = %.0f, Z = %.0f", mT.at<double>(0, 2), mT.at<double>(1, 0), 5000 - mT.at<double>(0, 0));
	string txt2 = format("Video size: %d x %d,  Runtime: %dms", img.cols, img.rows, tTime);

	putText(img, txt, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
	putText(img, txt1, Point(30, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
	putText(img, txt2, Point(30, 90), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);

}

void FillHoles(const Mat& i_in, Mat& i_out){
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(i_in, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	i_out.setTo(0);
	for (int i = 0; i < (int)contours.size(); i++)
	{
		drawContours(i_out, contours, i, Scalar(255), CV_FILLED, 8, hierarchy);
	}
}

void ComputeAngle(Point2f p1, Point2f p2, float& ang_rad, float& ang_deg)
{
	double de = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));

	if (de == 0) {
		ang_deg = 0; ang_rad = 0;
	}
	else
	{
		ang_rad = (float)acos((p2.y - p1.y) / de);
		if (p2.x < p1.x)
			ang_rad *= -1;
		ang_deg = ang_rad*180.f / (float)PI;
	}
}

void fprintf3Point(vector< vector< Point3f> > Points, string name)
{
	FILE * fp;
	fp = fopen(name.c_str(), "w");
	for (int i = 0; i < (int)Points.size(); ++i)
	{
		for (int j = 0; j < (int)Points[i].size(); ++j)
		{
			fprintf(fp, "%lf %lf %lf\n", Points[i][j].x, Points[i][j].y, Points[i][j].z);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


void fprintf2Point(vector< vector< Point2f> > Points, string name)
{
	FILE * fp;
	fp = fopen(name.c_str(), "w");
	for (int i = 0; i < (int)Points.size(); ++i)
	{
		for (int j = 0; j < (int)Points[i].size(); ++j)
		{
			fprintf(fp, "%lf %lf\n", Points[i][j].x, Points[i][j].y);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


void fprintfVectorMat(vector< Mat> matrix, string name)
{
	FILE * fp;
	fp = fopen(name.c_str(), "w");
	int i;
	printf("%s size %d, %d\n", name.c_str(), matrix.size(), matrix[0].cols, matrix[0].rows);
	for (i = 0; i < (int)matrix.size(); ++i)
	{
		for (int j = 0; j < matrix[i].rows; ++j)
		{
			for (int k = 0; k < matrix[i].cols; ++k)
			{
				fprintf(fp, "%lf ", matrix[i].at<  double >(j, k));
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}


	fclose(fp);
}

void fprintMatrix(Mat matrix, string name)
{
	FILE * fp;
	fp = fopen(name.c_str(), "w");
	int i, j;
	printf("%s size %d %d\n", name.c_str(), matrix.cols, matrix.rows);
	for (i = 0; i < matrix.rows; ++i)
	{
		for (j = 0; j < matrix.cols; ++j)
		{
			fprintf(fp, "%lf ", matrix.at<  double >(i, j));
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}

void drawDetectedCarpet(Mat& image, vector<Point2f>& imagePoints){

	for (int i = 0; i < (int)imagePoints.size(); i++){
		//string st = format("%d", i);
		//putText(image, st, Point(imagePoints[i].x+3, imagePoints[i].y+3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
		circle(image, imagePoints[i], 5, Scalar(0, 0, 255));
	}
}

void showRT(Mat& frame, Mat& mR, Mat& mT, int t){
	string txt = format("Rotation(degree) Yaw:%.0f, Pitch:%.0f, Roll:%.0f", mR.at<double>(0, 0)*180. / PI,
		mR.at<double>(1, 0)*180. / PI, mR.at<double>(2, 0)*180. / PI);
	string txt1 = format("Translation(mm) X:%.0f, Y:%.0f, Z:%.0f", mT.at<double>(0, 0), mT.at<double>(1, 0), mT.at<double>(2, 0));
	//string txt1 = format("Translation: X = %.0f, Y = %.0f, Z = %.0f", mT.at<double>(0, 2), mT.at<double>(1, 0), 5000 - mT.at<double>(0, 0));
	string txt2 = format("Video size: %d x %d,  Runtime: %dms", frame.rows, frame.cols, t);

	putText(frame, txt, Point(50, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	putText(frame, txt1, Point(50, 160), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	putText(frame, txt2, Point(50, 200), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
}