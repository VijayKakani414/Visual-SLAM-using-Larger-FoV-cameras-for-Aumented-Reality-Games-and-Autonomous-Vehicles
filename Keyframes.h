// based on ORB Slam2 paper: http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf

#ifndef KEYFRAMES_H
#define KEYFRAMES_H

#include <vector>
#include <list>
#include <set>
#include "Keyframe.h"
#include "Common.h"
#include "../DBoW2/DBoW2/BowVector.h"
#include "../DBoW2/DBoW2/TemplatedVocabulary.h"
#include "../DBoW2/DBoW2/FORB.h"

#include<mutex>

class Keyframe;

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

class Keyframes
{
public:
	Keyframes();

	static ORBVocabulary* m_pORBVocabulary;

	void add( Keyframe* pKF );
	void erase( Keyframe* pKF );
	void clear();

	// Relocalization
	bool Relocalize( Keyframe* pCurrentKeyframe );
	std::vector<Keyframe*> DetectRelocalizationChoices( Keyframe* pKeyframe );

protected:
	std::vector<Keyframe*> m_vKeyframes;
	std::vector<list<Keyframe*> > m_vInverseIndex;  // for every word in the vocabulary, list every keyframe containing it
	int m_nLastRelocFrameId;
	//std::mutex m_Mutex;
};

#endif  // KEYFRAMES_H