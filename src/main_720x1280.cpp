#include < stdio.h>  
#include < opencv2\opencv.hpp>  

#include "time.h"
#include < vector>  

#define PI 3.14159265359

using namespace std;
using namespace cv;

struct TrackPoint
{
	Point2f	imagePoint;
	Point2f imagePointNext;	// For estimation
	Point2f projectedPoint;	// Projected points after pose estimation
	Point3f	worldPoint;
	bool			bActive;
};

struct CameraParams
{
	Mat mIntrinsicMatrix;
	Mat mDistortionCoeffs;
	Vec3f rvec, tvec;
	Mat mR;
	Mat mT;
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
void InitializeTracker(CameraParams& camParams, vector<TrackPoint>& tracList, Mat& img, Mat& mGray);
void estimateCamPose(vector<TrackPoint>& trackList, CameraParams& camParams);

void validateTrackerList(vector<TrackPoint>& trackList, CameraParams& camParams, Mat& img, float fBorderGap);

double ED(Point2f p1, Point2f p2){
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}



void main()
{
	//////////////////////////////////////////////////////////////////////////////////////////////////  	
	string huanPath = "E:\\2016\\BENT\\Implementation\\Data\\Videos\\S5\\";
	vector<TrackPoint> trackList;
	CameraParams camParams;

	//if (0){		// Compute intrinsic matrix
	//	VideoCapture cap("20160111_150957.mp4");
	//	int nViews = 15;
	//	CameraIntrinsicCali(cap, nViews);
	//}

	//vector< vector< Point2f> > imagePointsSet;
	//vector< Point2f> imagePoints;
	//vector< Point2f> imagePointsNext;
	//vector< vector< Point3f> > objectPointsSet;
	//vector< Point3f> worldPoints;

	/// Load the intrinsic matrix and distortion coeffs if existed
	
	cv::FileStorage fileIntrinsicMatrix("IntrinsicMatrix.yml ", cv::FileStorage::READ);
	fileIntrinsicMatrix["Intrinsic matrix"] >> camParams.mIntrinsicMatrix;
	fileIntrinsicMatrix.release();

	
	cv::FileStorage fileDistortionCoeffs("DistortionCoeffs.yml", cv::FileStorage::READ);
	fileDistortionCoeffs["Distortion coeffs matrix"] >> camParams.mDistortionCoeffs;
	fileDistortionCoeffs.release();

	VideoCapture cap("0111_HD.avi"); 
	if (!cap.isOpened())  // check if we succeeded
		return;


	Mat mDisplay(1500, 2000, CV_8UC3);
	mDisplay.setTo(0);

	Mat frame;
	cap >> frame; // get a new frame from camera
	imshow("Pattern", frame);
	cvWaitKey(0);

	bool bFirst = true;
	Mat mGrayPrev, mGrayNext;

	generateWorldPointsTest(trackList);

	int nCnt = 0;

	for (;;)
	{
		frame;
		cap >> frame; // get a new frame from camera

		if (!frame.data)
			return;

		int t0 = clock();

		
		cvtColor(frame, mGrayNext, CV_BGR2GRAY);

		if (bFirst){	// Initialize the trackers
			bFirst = false;

			detectKeyPoints(frame, trackList);

			//RefineCorners(mGrayNext, trackList, 10);

			estimateCamPose(trackList, camParams);

			InitializeTracker(camParams, trackList, frame, mGrayNext);

			// Update the pose with more points
			estimateCamPose(trackList, camParams);

			int t = clock() - t0;

			mGrayPrev = mGrayNext.clone();

			showResult(camParams, trackList, frame, t, 0);
			imshow("Pattern", frame);
			cvWaitKey(0);
		}
		else
		{
			// Find back the points by optical flow
			opticalFlowTracker(mGrayPrev, mGrayNext, trackList);

			printf("Frame %d\n", nCnt);

			estimateCamPose(trackList, camParams);

			int t = clock() - t0;


			validateTrackerList(trackList, camParams, mGrayNext, 15);

			
						

			for (int i = 0; i < (int)trackList.size(); i++){
				trackList[i].imagePoint = trackList[i].imagePointNext;
			}

			mGrayNext.copyTo(mGrayPrev);

			showResult(camParams, trackList, frame, t, 1);
			imshow("Pattern", frame);
			cvWaitKey(10);

			nCnt++;
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////  
}

void validateTrackerList(vector<TrackPoint>& trackList, CameraParams& camParams, Mat& img, float fBorderGap){
	vector<Point2f> imagePts;
	vector<Point3f> worldPts;
	for (int i = 0; i < (int)trackList.size(); i++){
		imagePts.push_back(trackList[i].imagePoint);
		worldPts.push_back(trackList[i].worldPoint);
		}

	projectPoints(worldPts, camParams.mR, camParams.mT, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, imagePts);

	for (int i = 0; i < (int)trackList.size(); i++){
		trackList[i].projectedPoint = imagePts[i];
		// Check if the active point is near the border, then make it inactive
		if (trackList[i].bActive)
			if ((imagePts[i].x < fBorderGap) || (imagePts[i].x + fBorderGap > img.cols) || (imagePts[i].y < fBorderGap) || (imagePts[i].y + fBorderGap > img.rows))
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
	for (int i = 0; i < (int)trackList.size();i++)
		if (trackList[i].bActive){
		imagePts.push_back(trackList[i].imagePoint);
		worldPts.push_back(trackList[i].worldPoint);
		}
	if ((int)imagePts.size() < 4)
		return;

	solvePnP(worldPts, imagePts, camParams.mIntrinsicMatrix, camParams.mDistortionCoeffs, camParams.mR, camParams.mT);

}

void InitializeTracker(CameraParams& camParams, vector<TrackPoint>& tracList, Mat& img, Mat& mGray){

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
	RefineCorners(mGray, vImagePoints, 21);

	for (int i = 0; i < (int)vImagePoints.size(); i++)
		circle(img, vImagePoints[i], 3, Scalar(0, 0, 255));

	for (int i = 0; i < (int)vImagePoints.size(); i++)
	{
		tracList[vIdx[i]].worldPoint = vObjectPoints[i];
		tracList[vIdx[i]].imagePoint = vImagePoints[i];
		tracList[vIdx[i]].bActive = true;
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
		return Point2f(-1,-1);

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

	// The high table
	aPoint.worldPoint = (Point3f(2000, 0, 3000));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(1500, 0, 3000));
	trackList.push_back(aPoint);
	//vObjectPoints.push_back(Point3f(2000, 700, 3005));
	//vObjectPoints.push_back(Point3f(1495, 700, 3005));

	// The wood chair
	aPoint.worldPoint = (Point3f(1000, 0, 3000));
	trackList.push_back(aPoint);
	aPoint.worldPoint = (Point3f(530, 0, 3000));
	trackList.push_back(aPoint);

	//// The small white table
	//aPoint.worldPoint = (Point3f(1000, 0, 3000));
	//trackList.push_back(aPoint);
	//aPoint.worldPoint = (Point3f(530, 0, 3000));
	//trackList.push_back(aPoint);
	//aPoint.worldPoint = (Point3f(1000, 0, 3000));
	//trackList.push_back(aPoint);
	//aPoint.worldPoint = (Point3f(530, 0, 3000));
	//trackList.push_back(aPoint);
	
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

	m8U(Range(0, 120), Range(0, m8U.cols)).setTo(0);
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

	pt.y =  (float)(dCountourArea / dBoxArea);
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

	circle(img, vImagePoints[0], 5, Scalar(0, 0, 255),5);
	circle(img, vImagePoints[1], 3, Scalar(0, 0, 255),3);
	circle(img, vImagePoints[2], 3, Scalar(0, 0, 255), 3);
	circle(img, vImagePoints[3], 3, Scalar(0, 0, 255), 3);
	line(img, vImagePoints[0], vImagePoints[1], Scalar(0, 0, 255));
	line(img, vImagePoints[0], vImagePoints[2], Scalar(0, 0, 255));
	line(img, vImagePoints[0], vImagePoints[3], Scalar(0, 0, 255));

	if (flag){		// Show optical flow
		// Show the active tracking points
		for (int i = 0; i < (int)trackList.size();i++)
			if (trackList[i].bActive){
			string st = format("%d", i);
			putText(img, st, Point(trackList[i].imagePoint.x + 3, trackList[i].imagePoint.y + 3), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
			circle(img, trackList[i].imagePointNext, 3, Scalar(0, 0, 255));
			line(img, trackList[i].imagePoint, trackList[i].imagePointNext, Scalar(0, 0, 255));
			// Projected points
			circle(img, trackList[i].projectedPoint, 3, Scalar(0, 255, 0));
			}
	}

	string txt = format("Rotation(degree) Yaw:%.0f, Pitch:%.0f, Roll:%.0f", camParams.mR.at<double>(0, 0)*180. / PI,
		camParams.mR.at<double>(1, 0)*180. / PI, camParams.mR.at<double>(2, 0)*180. / PI);
	string txt1 = format("Translation(mm) X:%.0f, Y:%.0f, Z:%.0f", camParams.mT.at<double>(0, 0), camParams.mT.at<double>(1, 0), camParams.mT.at<double>(2, 0));
	//string txt1 = format("Translation: X = %.0f, Y = %.0f, Z = %.0f", mT.at<double>(0, 2), mT.at<double>(1, 0), 5000 - mT.at<double>(0, 0));
	string txt2 = format("Video size: %d x %d,  Runtime: %dms", img.rows, img.cols, tTime);

	putText(img, txt, Point(50, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	putText(img, txt1, Point(50, 160), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	putText(img, txt2, Point(50, 200), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	
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