#include <cv.h>
#include <highgui.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int main1()
{
	int numBoards = 20;
	int numCornersHor = 8;
	int numCornersVer = 6;
	float realBoardSz = 970.f/9;
	//float realBoardSz = 28.f;

	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);
	VideoCapture capture = VideoCapture("20160116_171404_cali_480.mp4");

	vector<vector<Point3f>> object_points;
	vector<vector<Point2f>> image_points;

	vector<Point2f> corners;
	int successes = 0;

	

	vector<Point3f> obj;
	for (int j = 0; j < numSquares; j++)
		obj.push_back(Point3d(realBoardSz*(float)(j / numCornersHor), realBoardSz * (float)(j%numCornersHor), 0.0f));


	Mat image;
	Mat gray_image;
	capture >> image;


	while (successes < numBoards)
	{
		cvtColor(image, gray_image, CV_BGR2GRAY);

		bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(image, board_sz, corners, found);
		}

		imshow("win1", image);
		//imshow("win2", gray_image);

		capture >> image;

		int key = waitKey(500);


		if (found != 0)
		{
			image_points.push_back(corners);
			object_points.push_back(obj);
			printf("Snap stored!\n");

			successes++;

			if (successes >= numBoards)
				break;

		}

		for (int i = 0; i < 40; i++)
			capture >> image;

	}

	Mat intrinsic = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;

	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;

	calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);

	// Save the intrinsic and distortion coeffs to file
	cv::FileStorage fileIntrinsicMatrix;
	fileIntrinsicMatrix.open("IntrinsicMatrix.yml", cv::FileStorage::WRITE);
	fileIntrinsicMatrix << "Intrinsic matrix" << intrinsic;
	fileIntrinsicMatrix.release();

	cv::FileStorage fileDistortionCoeffs;
	fileDistortionCoeffs.open("DistortionCoeffs.yml", cv::FileStorage::WRITE);
	fileDistortionCoeffs << "Distortion coeffs matrix" << distCoeffs;
	fileDistortionCoeffs.release();

	Mat imageUndistorted;
	while (1)
	{
		capture >> image;
		if (image.rows == 0)
			return false;

		undistort(image, imageUndistorted, intrinsic, distCoeffs);

		imshow("win1", image);
		imshow("win2", imageUndistorted);

		waitKey(1);
	}

	capture.release();

	return 0;
}