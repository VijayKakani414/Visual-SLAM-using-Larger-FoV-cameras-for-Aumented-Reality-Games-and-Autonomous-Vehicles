// original source of this file shield_slam - https://github.com/MohitShridhar/shield_slam/blob/master/shield_slam/ORB.cpp

#ifndef COMMON_H
#define COMMON_H

#include <opencv2/opencv.hpp>
#include <cassert>
#include <functional>
#include <memory>

typedef std::vector<cv::KeyPoint> KeypointArray;
typedef std::vector<cv::Point2f> PointArray;

extern Mat camera_matrix, dist_coeff, img_size;

struct Pose {
	Point3f m_p3fPos;
	Point3f m_p3fRot;
};

#endif // COMMON_H
