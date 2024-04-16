// original source of this file shield_slam - https://github.com/MohitShridhar/shield_slam/blob/master/shield_slam/ORB.cpp
// extended using ORB Slam2 paper: http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf
// for more info on the Bag Of Words:  http://doriangalvez.com/papers/GalvezTRO12.pdf

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "../DBoW2/DBoW2/BowVector.h"
#include "../DBoW2/DBoW2/FeatureVector.h"
#include "Common.h"
#include "Keyframes.h"
#include <mutex>

class DescriptedPC;

class Keyframe
{
public:
	Keyframe();
	Keyframe( DescriptedPC* pPointCloud );
	virtual ~Keyframe() {}

	DescriptedPC* m_pPointCloud;

	//BoW
	DBoW2::BowVector m_BowVector;
	DBoW2::FeatureVector m_FeatureVector;

	// Members used by the keyframes class for relocalization
	long unsigned int m_nRelocQuery;
	int m_nRelocWords;
	float m_fRelocScore;
	long unsigned int m_nId;
	static long unsigned int m_nNextId;

	vector<Keyframe*> GetBestCovisibilityKeyframes( const int &number );
	void UpdateCovisibleKeyframes();
	void UpdateConnections();
	void AddConnection( Keyframe* pKeyFrame, int weight );
	void AddChild( Keyframe* pKeyframe );
	void ComputeBagOfWords();
	
	// These are required to support the network of co-visible keyframes as described in the orb slam paper
	std::vector<Keyframe*> m_vCoVisibleKeyframes;
	std::vector<int> m_vCoVisibleKeyframeWeights;
	std::map<Keyframe*, int> m_mapKeyframeToWeight;
	//std::vector<Keyframe*> m_vEssentialKeyframes;  // can probably just grab the first, or first and second to create an Essential graph

	// loop edges
	bool m_bFirstConnection;
	Keyframe* m_pParent;
	std::set<Keyframe*> m_setChildren;
	std::set<Keyframe*> m_setLoopEdges;

	// pointers to the MapPoint data
	// vector<MapPoint*> m_vpMapPoints;
};

#endif // KEYFRAME_H