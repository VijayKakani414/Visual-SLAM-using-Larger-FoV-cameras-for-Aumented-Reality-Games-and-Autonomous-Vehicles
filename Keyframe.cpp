// created based on ORB Slam2 paper: http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf

#include "Keyframe.h"
#include "DescriptedPC.h"

long unsigned int Keyframe::m_nNextId;

Keyframe::Keyframe()
{
	m_nId = m_nNextId++;
}

Keyframe::Keyframe( DescriptedPC* pPointCloud )
{
	m_nId = m_nNextId++;
	m_pPointCloud = pPointCloud;

	ComputeBagOfWords();
}

vector<Keyframe*> Keyframe::GetBestCovisibilityKeyframes( const int &number )
{
	// unique_lock<mutex> lockKey( mMutexConnections );

	if ((int)m_vCoVisibleKeyframes.size() < number)  // if list is shorter than requested, return them all
		return m_vCoVisibleKeyframes;
	else  // otherwise return top requested number of keyframes
		return vector<Keyframe*>( m_vCoVisibleKeyframes.begin(), m_vCoVisibleKeyframes.begin() + number );
}

void Keyframe::UpdateCovisibleKeyframes()
{
	//unique_lock<mutex> lock( mMutexConnections );
	vector<pair<int, Keyframe*> > vPairs;
	vPairs.reserve( m_vCoVisibleKeyframes.size() );
	for (map<Keyframe*, int>::iterator mit = m_mapKeyframeToWeight.begin(), mend = m_mapKeyframeToWeight.end(); mit != mend; mit++)
		vPairs.push_back( make_pair( mit->second, mit->first ) );

	sort( vPairs.begin(), vPairs.end() );
	list<Keyframe*> lKFs;
	list<int> lWs;
	for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
	{
		lKFs.push_front( vPairs[i].second );
		lWs.push_front( vPairs[i].first );
	}

	m_vCoVisibleKeyframes = vector<Keyframe*>( lKFs.begin(), lKFs.end() );
	m_vCoVisibleKeyframeWeights = vector<int>( lWs.begin(), lWs.end() );
}

void Keyframe::UpdateConnections()  // todo - revise
{
	map<Keyframe*, int> mapKeyframeToCount;

	// unique_lock<mutex> lockMPs( mMutexFeatures );

	// For all map points in keyframe check in which other keyframes are they seen and count them
	for (size_t i = 0; i < m_pPointCloud->vecPoint3D.size(); i++)
	{
		//if (m_pPointCloud->m_vBoolIsBad[i])
		//	continue;

		map<Keyframe*, size_t> observations = m_pPointCloud->m_vMapKeyframeToIndexInKeyframe[i];

		for (map<Keyframe*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			if (mit->first->m_nId == m_nId)
				continue;
			mapKeyframeToCount[mit->first]++;
		}
	}

	int nmax = 0;
	Keyframe* pKFmax = NULL;
	int th = 15;

	vector<pair<int, Keyframe*> > vPairs;
	vPairs.reserve( mapKeyframeToCount.size() );
	for (map<Keyframe*, int>::iterator mit = mapKeyframeToCount.begin(), mend = mapKeyframeToCount.end(); mit != mend; mit++)
	{
		if (mit->second > nmax)
		{
			nmax = mit->second;
			pKFmax = mit->first;
		}
		// if the counter is greater than threshold add connection
		if (mit->second >= th)
		{
			vPairs.push_back( make_pair( mit->second, mit->first ) );
			(mit->first)->AddConnection( this, mit->second );
		}
	}

	// if no keyframe counter is over the threshold add the one with biggest count
	if (vPairs.empty())
	{
		vPairs.push_back( make_pair( nmax, pKFmax ) );
		pKFmax->AddConnection( this, nmax );
	}

	sort( vPairs.begin(), vPairs.end() );
	list<Keyframe*> lKFs;
	list<int> lWs;
	for (size_t i = 0; i < vPairs.size(); i++)
	{
		lKFs.push_front( vPairs[i].second );
		lWs.push_front( vPairs[i].first );
	}

	{
		// unique_lock<mutex> lockCon( m_mutexConnections );

		m_mapKeyframeToWeight = mapKeyframeToCount;
		m_vCoVisibleKeyframes = vector<Keyframe*>( lKFs.begin(), lKFs.end() );
		m_vCoVisibleKeyframeWeights = vector<int>( lWs.begin(), lWs.end() );

		if (m_bFirstConnection && m_nId != 0)
		{
			m_pParent = m_vCoVisibleKeyframes.front();
			m_pParent->AddChild( this );
			m_bFirstConnection = false;
		}
	}
}

void Keyframe::AddConnection( Keyframe* pKeyFrame, int weight )
{
	// unique_lock<mutex> lock( m_mutexConnections );
	if (m_mapKeyframeToWeight.count( pKeyFrame ) == 0 || m_mapKeyframeToWeight[pKeyFrame] != weight)
	{
		m_mapKeyframeToWeight[pKeyFrame] = weight;
		UpdateCovisibleKeyframes();
	}
}

void Keyframe::AddChild( Keyframe* pKeyframe )
{
	// unique_lock<mutex> lock( m_mutexConnections );
	m_setChildren.insert( pKeyframe );
}

void Keyframe::ComputeBagOfWords()
{
	assert( m_pPointCloud );

	if (m_BowVector.empty() || m_FeatureVector.empty())
	{
		vector<cv::Mat> vCurrentDesc;  // it's a shame that we have to copy the descriptors, but the BoW library takes a vector<cv::Mat>
		vCurrentDesc.reserve( m_pPointCloud->matDescriptors.rows );
		for (int r = 0; r < m_pPointCloud->matDescriptors.rows; r++)
			vCurrentDesc.push_back( m_pPointCloud->matDescriptors.row(r) );

		Keyframes::m_pORBVocabulary->transform( vCurrentDesc, m_BowVector, m_FeatureVector, 4 );  // this works for a tree which has 6 levels
	}
}

