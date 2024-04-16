// based on ORB Slam2 paper: http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf

#include "Keyframes.h"
#include "Keyframe.h"
#include<mutex>
#include <direct.h>

using namespace std;

ORBVocabulary* Keyframes::m_pORBVocabulary;

Keyframes::Keyframes()
{
	// load the ORB vocabulary from binary file
	clock_t tStart = clock();
	m_pORBVocabulary = new ORBVocabulary();

	char buffer[100];
	_getcwd( buffer, 100 );

	std::string strVocFile = "..\\ORBVoc.bin";
	bool bVocLoaded = m_pORBVocabulary->loadFromBinaryFile( strVocFile );

	//std::string strVocFile = "C:\\Users\\Bob\\Documents\\Code\\bentdev_visionin-c\\Phase 3\\ORBVoc.txt";
	//bool bVocLoaded = m_pORBVocabulary->loadFromTextFile( strVocFile );

	if (!bVocLoaded)
	{
		cerr << "Failed to open: " << strVocFile << endl;
		exit( -1 );
	}
	printf( "Vocabulary loaded in %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC );

	m_vInverseIndex.resize( m_pORBVocabulary->size() );
}

void Keyframes::add( Keyframe *pKeyframe )
{
	//unique_lock<mutex> lock( m_Mutex );

	for (DBoW2::BowVector::const_iterator it = pKeyframe->m_BowVector.begin(), vend = pKeyframe->m_BowVector.end(); it != vend; it++)
		m_vInverseIndex[it->first].push_back( pKeyframe );

	pKeyframe->UpdateCovisibleKeyframes();
}

void Keyframes::erase( Keyframe* pKeyframe )
{
	//unique_lock<mutex> lock( m_Mutex );

	// Erase elements in the Inverse that match the entry
	for (DBoW2::BowVector::const_iterator vit = pKeyframe->m_BowVector.begin(), vend = pKeyframe->m_BowVector.end(); vit != vend; vit++)
	{
		// List of keyframes that share the word
		list<Keyframe*> &listKeyframes = m_vInverseIndex[vit->first];

		for (list<Keyframe*>::iterator lit = listKeyframes.begin(), lend = listKeyframes.end(); lit != lend; lit++)
		{
			if (pKeyframe == *lit)
			{
				// lit->UpdateCovisibleKeyframes();  // todo
				listKeyframes.erase( lit );
				break;
			}
		}
	}
}

void Keyframes::clear()
{
	m_vInverseIndex.clear();
	m_vInverseIndex.resize( m_pORBVocabulary->size() );
}

vector<Keyframe*> Keyframes::DetectRelocalizationChoices( Keyframe *pMatchKeyframe )
{
	list<Keyframe*> listKeyframesSharingWords;

	// Search the keyframes that have a common word with the current frame
	{
		//unique_lock<mutex> lock( m_Mutex );

		for (DBoW2::BowVector::const_iterator bowit = pMatchKeyframe->m_BowVector.begin(), vend = pMatchKeyframe->m_BowVector.end(); bowit != vend; bowit++)
		{
			list<Keyframe*> &listKeyframes = m_vInverseIndex[bowit->first];

			for (list<Keyframe*>::iterator listit = listKeyframes.begin(), lend = listKeyframes.end(); listit != lend; listit++)
			{
				Keyframe* pKeyframe = *listit;
				if (pKeyframe->m_nRelocQuery != pMatchKeyframe->m_nId)
				{
					pKeyframe->m_nRelocWords = 0;
					pKeyframe->m_nRelocQuery = pMatchKeyframe->m_nId;
					listKeyframesSharingWords.push_back( pKeyframe );
				}
				pKeyframe->m_nRelocWords++;
			}
		}
	}
	if (listKeyframesSharingWords.empty())
		return vector<Keyframe*>();

	// Compare against keyframes if they share a count of words above some threshold (80% of max)
	int maxCommonWords = 0;
	for (list<Keyframe*>::iterator listit = listKeyframesSharingWords.begin(), lend = listKeyframesSharingWords.end(); listit != lend; listit++)
	{
		if ((*listit)->m_nRelocWords > maxCommonWords)
			maxCommonWords = (*listit)->m_nRelocWords;
	}

	int minCommonWords = (int)(maxCommonWords*0.8f);

	list<pair<float, Keyframe*> > listPairScoreMatch;

	int nScores = 0;

	// Calculate a score based on similarity
	for (list<Keyframe*>::iterator listit = listKeyframesSharingWords.begin(), lend = listKeyframesSharingWords.end(); listit != lend; listit++)
	{
		Keyframe* pKeyframe = *listit;

		if (pKeyframe->m_nRelocWords > minCommonWords)
		{
			nScores++;
			float si = (float)m_pORBVocabulary->score( pMatchKeyframe->m_BowVector, pKeyframe->m_BowVector );
			pKeyframe->m_fRelocScore = si;
			listPairScoreMatch.push_back( make_pair( si, pKeyframe ) );
		}
	}

	if (listPairScoreMatch.empty())
		return vector<Keyframe*>();

	list<pair<float, Keyframe*> > listPairAccumulatedScoreMatch;
	float bestAccScore = 0;

	// Accumulate score based on covisibility
	for (list<pair<float, Keyframe*> >::iterator it = listPairScoreMatch.begin(), itend = listPairScoreMatch.end(); it != itend; it++)
	{
		Keyframe* pKeyframe = it->second;
		vector<Keyframe*> vpNeighs; // = pKeyframe->GetBestCovisibilityKeyframes( 10 );

		float bestScore = it->first;
		float accScore = bestScore;
		Keyframe* pBestKF = pKeyframe;
		for (vector<Keyframe*>::iterator vKFIterator = vpNeighs.begin(), vend = vpNeighs.end(); vKFIterator != vend; vKFIterator++)
		{
			Keyframe* pKeyframe2 = *vKFIterator;
			if (pKeyframe2->m_nRelocQuery != pMatchKeyframe->m_nId)
				continue;

			accScore += pKeyframe2->m_fRelocScore;
			if (pKeyframe2->m_fRelocScore > bestScore)
			{
				pBestKF = pKeyframe2;
				bestScore = pKeyframe2->m_fRelocScore;
			}

		}
		listPairAccumulatedScoreMatch.push_back( make_pair( accScore, pBestKF ) );
		if (accScore > bestAccScore)
			bestAccScore = accScore;
	}

	// Return all keyframes with a score higher than some threshold percentage of the maximum score
	float minScoreToRetain = 0.75f*bestAccScore;
	set<Keyframe*> setAlreadyAddedKF;
	vector<Keyframe*> vKFRelocCandidates;
	vKFRelocCandidates.reserve( listPairAccumulatedScoreMatch.size() );
	for (list<pair<float, Keyframe*> >::iterator it = listPairAccumulatedScoreMatch.begin(), itend = listPairAccumulatedScoreMatch.end(); it != itend; it++)
	{
		const float &si = it->first;
		if (si > minScoreToRetain)
		{
			Keyframe* pKFi = it->second;
			if (!setAlreadyAddedKF.count( pKFi ))
			{
				vKFRelocCandidates.push_back( pKFi );
				setAlreadyAddedKF.insert( pKFi );
			}
		}
	}

	return vKFRelocCandidates;
}

bool Keyframes::Relocalize( Keyframe* pCurrentKeyframe )
{
	pCurrentKeyframe->ComputeBagOfWords();

	// tracking is lost
	// get keyframe choices based on BoW
	vector<Keyframe*> vecKeyframeChoices = DetectRelocalizationChoices( pCurrentKeyframe );

	if (vecKeyframeChoices.empty())
		return false;

	const int numKeyframes = vecKeyframeChoices.size();

	int nCandidates = 0;

	for (int i = 0; i < numKeyframes; i++)
	{
		Keyframe* pKF = vecKeyframeChoices[i];

		// Perform some iterations of PnP RANSAC
		// Until we find a camera pose supported by enough inliers

		// Perform 5 Ransac Iterations

		// If Ransac reaches max. iterations discard that keyframe, and continue to next one

		// If a Camera Pose is computed, optimize it with bundle adjustment

		// If few inliers, (< 50) search by projection in a coarse window and optimize again
		// If many inliers, but still not enough, (inliers>30 && inliers<50) search by projection again in a narrower window
		// (the camera has by now been optimized with many points)
		// Final optimization

		// If the pose is supported by enough inliers (>50) stop ransacs and break out
	}

	m_nLastRelocFrameId = pCurrentKeyframe->m_nId;
	return true;
}
