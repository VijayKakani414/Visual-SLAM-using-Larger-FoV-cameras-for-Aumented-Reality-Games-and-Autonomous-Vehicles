#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

class Keyframe;
using namespace cv;
using namespace std;

class MapPoint
{
public:
	void SetPoint3D( Point3f coord ) { point_3D = coord; }
	Point3f GetPoint3D( void ) { return point_3D; }

	void SetPoint2D( Point2f coord ) { point_2D = coord; }
	Point2f GetPoint2D( void ) { return point_2D; }

	void SetDesc( Mat& desc ) { descriptor = desc.clone(); }
	Mat GetDesc( void ) { return descriptor; }
	bool isBad() { return m_bIsBad; }
	map<Keyframe*, size_t> GetObservations()
	{
		// unique_lock<mutex> lock( m_mutexFeatures );
		return m_mapKeyframeToIndexInKeyframe;
	}

protected:
	Point2f point_2D;
	Point3f point_3D;
	Mat descriptor;
	std::map<Keyframe*, size_t> m_mapKeyframeToIndexInKeyframe;  //  observations

	bool m_bIsBad;

	// std::mutex m_mutexFeatures;
};

#endif // MAPPOINT_H