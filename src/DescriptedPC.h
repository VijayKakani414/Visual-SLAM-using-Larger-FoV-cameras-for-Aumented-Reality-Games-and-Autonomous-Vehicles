#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>
#include <stdlib.h>

class DescriptedPC 
{
public:
	std::vector<cv::Point3d> vecPoint3D;						///< Point cloud
	cv::Mat				matDescriptors;
	int				nDecsMethod;
	cv::Mat				mR;
	cv::Mat				mT;

	// additions to support BOW

	// the following vector of maps permits looking up the index where a particular point exists in another keyframe.
	// used to build the co-visibility graph

	//vector<map<Keyframe*, size_t>> m_vMapKeyframeToIndexInKeyframe;  // observations - for each point, if it exists in a keyframe, what index is it stored at?

	// vector<bool> m_vBoolIsBad;  // this permits marking a point bad instead of actually deleting, which require re-allocation
};
