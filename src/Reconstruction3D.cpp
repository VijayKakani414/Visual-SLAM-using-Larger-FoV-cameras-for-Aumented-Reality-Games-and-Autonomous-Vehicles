#include "Reconstruction3D.h"

#define EPSILON 0.001
#define PI 3.14159265

CReconstruction3D::CReconstruction3D()
{
	m_bSaveCloud = false;
	m_bLoadCloud = false;
}

CReconstruction3D::~CReconstruction3D()
{
}

void CReconstruction3D::saveGeometry(vector<pcl::PolygonMesh>& vecTriangulates){
	//FileStorage fs("Geometry.yml", FileStorage::WRITE);
	//Mat mVerts, mFaces;
	//int nVertices;
	//for (int i = 0; i < vecTriangulates.size(); i++){
	//	cvtVec2Mat(algParams.vecClusteredPoints3D[i], mVerts);
	//	mVerts = mVerts.t();
	//	nVertices = vecTriangulates[i].polygons.size();
	//	mFaces.create(nVertices, 3, CV_64F);
	//	for (int j = 0; j < nVertices; j++){
	//		mFaces.at<double>(j, 0) = vecTriangulates[i].polygons[j].vertices[0];
	//		mFaces.at<double>(j, 1) = vecTriangulates[i].polygons[j].vertices[1];
	//		mFaces.at<double>(j, 2) = vecTriangulates[i].polygons[j].vertices[2];
	//	}
	 
	//	fs << "Vertices" << mVerts << "Faces" << mFaces;
	//}
	//fs.release();
}

void CReconstruction3D::initParams(int nWidth, int nHeight){
	// Parameter values for tracking
	algParams.movingParams.nPoints = 500;		// Number of maximum points
	algParams.movingParams.dQuality = 0.02;
	algParams.movingParams.dMinDistance = 6;
	algParams.movingParams.fErr = 10;
	algParams.movingParams.nBlSize = 5;
	algParams.movingParams.nWinSize = 25;
	// Parameter values for computing the fundamental matrix
	m_sFundamentalParams.nPoints = 600;
	m_sFundamentalParams.dQuality = 0.02;
	m_sFundamentalParams.dMinDistance = 5;
	m_sFundamentalParams.fErr = 20;
	m_sFundamentalParams.nBlSize = 5;
	m_sFundamentalParams.nWinSize = 21;
	// Parameter values for detect points for reconstruction
	m_sReconstructParams.nPoints = 600;
	m_sReconstructParams.dQuality = 0.02;
	m_sReconstructParams.dMinDistance = 5;
	m_sReconstructParams.fErr = 1;
	m_sReconstructParams.nBlSize = 9;
	m_sReconstructParams.nWinSize = 21;

	//// Parameter values for tracking
	//algParams.movingParams.nPoints = 500;		// Number of maximum points
	//algParams.movingParams.dQuality = 0.01;
	//algParams.movingParams.dMinDistance = 6;
	//algParams.movingParams.fErr = 10;
	//algParams.movingParams.nBlSize = 5;
	//algParams.movingParams.nWinSize = 25;
	//// Parameter values for computing the fundamental matrix
	//m_sFundamentalParams.nPoints = 600;
	//m_sFundamentalParams.dQuality = 0.01;
	//m_sFundamentalParams.dMinDistance = 5;
	//m_sFundamentalParams.fErr = 20;
	//m_sFundamentalParams.nBlSize = 5;
	//m_sFundamentalParams.nWinSize = 21;
	//// Parameter values for detect points for reconstruction
	//m_sReconstructParams.nPoints = 600;
	//m_sReconstructParams.dQuality = 0.01;
	//m_sReconstructParams.dMinDistance = 5;
	//m_sReconstructParams.fErr = 1;
	//m_sReconstructParams.nBlSize = 5;
	//m_sReconstructParams.nWinSize = 21;

	m_dMeanThresh = 25;
	m_dStdThresh = 8;
	m_nImgHeight = nHeight;
	m_nImgWidth = nWidth;
	m_mROI.create(nHeight, nWidth, CV_8U);
	m_mROI.setTo(0);
	m_mROI(Range(32, m_nImgHeight-32), Range(32, m_nImgWidth-32)).setTo(255);

	// Initialize the descriptors
	m_pDescDetector = GFTTDetector::create(500, 0.01, 8, 5);
	m_pDescExtractor = ORB::create();
	//m_pDescExtractor = KAZE::create();
	m_pDescMatcher = BFMatcher::create("BruteForce");

	algParams.nStatus = NO_ERR;
	createColors(m_vecColors);
}

void CReconstruction3D::removeAbnormalPoints3D(vector<Point3d>& vecInput, vector<Point3d>& vecOutput){
	Mat mData3D;
	cvtVec2Mat(vecInput, mData3D);
	mData3D = mData3D.t();
	int nDims = mData3D.cols;
	double h = 0;
	double nSamples = mData3D.rows;

	// Remove the points that are obviously abnormal
	Mat mPointNorm(mData3D.rows, 1, CV_64FC1), mSorted;
	for (int i = 0; i < nSamples; i++)
		mPointNorm.at<double>(i, 0) = norm(mData3D.row(i));

	cv::sort(mPointNorm, mSorted, CV_SORT_EVERY_COLUMN);
	double dThresh = 3 * mSorted.at<double>(nSamples / 2, 0);
	mPointNorm = mPointNorm < dThresh;

	vecOutput = vecInput;

	if (sum(mPointNorm).val[0] / 255 == nSamples)
		return;

	vector<Point3d> output;
	output.clear();
	Point3d point;
	output.resize(nSamples);
	mPointNorm.convertTo(mPointNorm, CV_32F);
	for (int i = nSamples - 1; i >= 0; i--)
		if (mPointNorm.at<float>(i, 0) == 0)
			vecOutput.erase(vecOutput.begin() + i);

}

bool CReconstruction3D::findFirstSeedFrame(Mat & image){
	algParams.img1 = image;
	/// Parameters for Shi-Tomasi algorithm
	cvtColor(algParams.img1, m_mGrayImg1, CV_RGB2GRAY);
	float df_score = ComputeDFScore2(m_mGrayImg1);	// Defocus score
	cout << df_score << endl;
	if (df_score < m_fDefocusScoreThresh)
		return false;

	
	bool b = detectSeedPoints(algParams.movingParams.nPoints, algParams.movingParams.dQuality, algParams.movingParams.dMinDistance,
		algParams.movingParams.nBlSize, true);//wrong
	if (b){
		cvtColor(image, m_mGrayImg1, CV_RGB2GRAY);
	}
	return b;
}

void CReconstruction3D::readInfo(string intrinsicPath, string coeffPath){
	cv::FileStorage fs;
	fs = FileStorage(intrinsicPath, cv::FileStorage::READ);
	if (fs.isOpened()){
		fs["Intrinsic matrix"] >> algParams.intrinsic;
	}
	else
	{
		algParams.nStatus = ERR_NO_CAM_INTRINSIC;
		fs.release();
		return;
	}
	fs = FileStorage(coeffPath, cv::FileStorage::READ);
	if (fs.isOpened())
		fs["Distortion coeffs matrix"] >> algParams.distcoeff;
	else{
		algParams.nStatus = ERR_NO_CAM_DISTORTION;
		fs.release();
	}
}

double CReconstruction3D::estimateMSBandwidth(const Mat& mData3D){
	int nDims = mData3D.cols;
	double h = 0;
	double nSamples = mData3D.rows;

	// Remove the points that are obviously abnormal
	Mat mPointNorm(mData3D.rows, 1, CV_64FC1), mSorted;
	for (int i = 0; i < nSamples; i++)
		mPointNorm.at<double>(i, 0) = norm(mData3D.row(i));

	cv::sort(mPointNorm, mSorted, CV_SORT_EVERY_COLUMN);
	double dThresh = 2 * mSorted.at<double>(nSamples / 2, 0);
	mPointNorm = mPointNorm < dThresh;

	// Estimate the bandwidth using Silverman method
	for (int i = 0; i < nDims; i++){
		Mat& mCol = mData3D.col(i);
		Scalar scMean, scStd;
		meanStdDev(mCol, scMean, scStd, mPointNorm);
		//meanStdDev(mCol, scMean, scStd);
		h += 1.06*scStd.val[0] * pow(nSamples, -1 / 5);
	}
	h /= nDims;
	return h;
}


double CReconstruction3D::ED(const vector<double> &point_a, const vector<double> &point_b){
	double total = 0;
	for (int i = 0; i < (int)point_a.size(); i++){
		total += (point_a[i] - point_b[i]) * (point_a[i] - point_b[i]);
	}
	return sqrt(total);
}

double CReconstruction3D::ED(const Point2d &point_a, const Point2d &point_b){
	double total = 0;
	total = pow((point_a.x - point_b.x), 2) + pow((point_a.y - point_b.y), 2);
	return sqrt(total);
};

double CReconstruction3D::ED(const Point3d &point_a, const Point3d &point_b){
	double total = 0;
	total = pow((point_a.x - point_b.x), 2) + pow((point_a.y - point_b.y), 2) + pow((point_a.z - point_b.z), 2);
	return sqrt(total);
};


//void CReconstruction3D::registerPointCloud(int nWinSize, float er){
//	int nLen = (int)algParams.vecPoints3D.size();
//	if (nLen <= 0) //if there is nothing to register, exit
//		return;
//
//	if (nLen == 1){//if there is only one point cloud to register, what are you doing? Put the point-cloud in the proper place for it to be visualized
//		algParams.vecRegisteredPoints3D.push_back(algParams.vecPoints3D[0]);
//		for (int i = 0; i < m_vImgPointsInlier2.size(); i++)
//			m_vecSeed2D4Affine.push_back(m_vImgPointsInlier2[i]);//unsure what this is about --Oliver
//		return;
//	}
//
//	// Find the affine transform from the previous view pair to the current
//	vector<uchar> status;
//	vector<float> err;
//	vector<Point2f> nextPoints2D;
//	Size winSize(nWinSize, nWinSize);
//	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.001);
//	calcOpticalFlowPyrLK(m_mGrayImg1, m_mGrayImg2, m_vecSeed2D4Affine, nextPoints2D, status, err, winSize, 4, termcrit, 0);
//
//	Mat canvas;
//	// Found optical flows
//	vector<Point2f> prevPoints, nextPoints;
//	vector<Point3d>& vec3Ds = algParams.vecPoints3D[nLen - 2];
//	vector<Point3d> prevPoints3D;
//	for (int i = 0; i < nextPoints2D.size(); i++){
//		if (status[i] && (err[i] < er)){//check to see if the optical point exists in both frames, and if it is calculated well.
//			prevPoints.push_back(m_vecSeed2D4Affine[i]);
//			nextPoints.push_back(nextPoints2D[i]);
//			prevPoints3D.push_back(vec3Ds[i]);//okay, so this is just the 3D points we calculated on the previous frame-pair
//		}
//	}
//
//
//	undistortPoints(prevPoints, prevPoints, algParams.intrinsic, algParams.distcoeff);
//	undistortPoints(nextPoints, nextPoints, algParams.intrinsic, algParams.distcoeff);
//
//	Mat matReconstruct3D;
//	Mat tmpImg;
//    //nextPoints calculated purely from previous points and optical flow. D:
//	triangulatePoints(algParams.P, algParams.P1, prevPoints, nextPoints, matReconstruct3D);//matReconstruct 3D is the triangulated points
//	tmpImg = matReconstruct3D.row(3).clone(); //First, take the fourth row of matReconstruct3D
//	vconcat(tmpImg, tmpImg, tmpImg); //Second, double it
//	vconcat(tmpImg, tmpImg, tmpImg); //Then double that double (four identical rows now?)
//	divide(matReconstruct3D, tmpImg, matReconstruct3D);//Okay, so we divide the matReconstruct3D by its fourth row? Unsure why/what this does.
//
//    vector<Point3d> nextPoints3D;
//	vector<uchar> inliers3D;
//	cvtMat32F2Vec3D(matReconstruct3D, nextPoints3D);//okay, so this is filling the nextPoints3D vector
//    //with points that are stored in the first three rows of the matReconstruct3D matrix
//    
//	Mat affineMat;
//    //prevPoints3D is just the point cloud points calculated in the last frame-pair (The first frame and the 7th frame)
//    //nextPoints3D is the point cloud points calculated in this frame-pair? (the 7th frame and the... 60th?)
//	estimateAffine3D(nextPoints3D, prevPoints3D, affineMat, inliers3D, 3);//This requires the same number of points for both vectors
//    //it also requires each point to correspond, one-to-one with the corresponding point in the other vector
//	m_vecAffineMat.push_back(affineMat);
//
//	Mat R, T;
//	Mat tmp;
//	//// Compute the transformation error
//	//cvtVec2Mat(prevPoints3D, tmp);
//	//R = affineMat(Range(0, 3), Range(0, 3)).clone();
//	//T = affineMat.col(3);
//	//tmp = R*tmp;
//	//for (int i = 0; i < tmp.cols; i++){
//	//	tmp.col(i) = tmp.col(i) + T;
//	//}
//	//tmp.convertTo(tmp, CV_32F);
//	//tmp = tmp - matReconstruct3D(Range(0, 3), Range(0, matReconstruct3D.cols));
//	//tmp = tmp.mul(tmp);
//	//double dErr = sqrt(sum(tmp).val[0])/matReconstruct3D.cols;
//	//cout << dErr;
//
//	// Transformation
//	vector<Point3d>& vecCurrent3Ds = algParams.vecPoints3D[nLen - 1];
//	
//	cvtVec2Mat(vecCurrent3Ds, tmp);
//
//	for (int k = nLen - 2; k >= 0; k--){ //go back all the way, using the calculated affine transforms, to convert the points into the original frame of reference
//		R = m_vecAffineMat[k](Range(0, 3), Range(0, 3)).clone();
//		T = m_vecAffineMat[k].col(3);
//		tmp = R*tmp;
//		for (int i = 0; i < tmp.cols; i++){
//			tmp.col(i) = tmp.col(i) + T;
//		}
//	}
//
//	vector<Point3d> vecTransformed;
//	cvtMat2Vec(tmp, vecTransformed);
//	algParams.vecRegisteredPoints3D.push_back(vecTransformed);
//
//
//	//////////////////////////////////////////////////////////////////////////
//	m_vecSeed2D4Affine.clear();
//	m_vecSeed2D4Affine.resize((int)m_vImgPointsInlier2.size());
//	for (int i = 0; i < m_vImgPointsInlier2.size(); i++)
//		m_vecSeed2D4Affine[i] = m_vImgPointsInlier2[i];
//}

Mat CReconstruction3D::calculateProjectionMatrix(vector<Point3d>  prevPoints3D, vector<Point2f>  nextPoints){
	
	Mat projectionInliers;
	Mat cameraMatrix;
	Mat rotMatrix;

	Mat rotationVector;
	Mat translationVector;

	pcl::PointCloud<pcl::PointXYZ>::Ptr nextPointsPC(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i < nextPoints.size(); i++){
		pcl::PointXYZ temp;
		temp.x = nextPoints[i].x;
		temp.y = nextPoints[i].y;
		temp.z = 0.0;
		nextPointsPC->push_back(temp);
	}
	pcl::SampleConsensusModelRegistration<pcl::PointXYZ>::Ptr model(new pcl::SampleConsensusModelRegistration<pcl::PointXYZ>(nextPointsPC));
	vector<Point2f> inlierPoints = nextPoints;
	vector<Point3d> inliers3D = prevPoints3D;
	int previous_inliers = 0;
	for (int i = 0; i < 100; i++){

		solvePnP(inliers3D, inlierPoints, algParams.intrinsic, algParams.distcoeff, rotationVector, translationVector);
		vector<Point2d> imagepoints;
		projectPoints(prevPoints3D, rotationVector, translationVector, algParams.intrinsic, algParams.distcoeff, imagepoints); 
		pcl::PointCloud<pcl::PointXYZ>::Ptr imagepointsPC(new pcl::PointCloud<pcl::PointXYZ>());
		for (int j = 0; j < imagepoints.size(); j++){
			pcl::PointXYZ temp;
			temp.x = imagepoints[j].x;
			temp.y = imagepoints[j].y;
			temp.z = 0.0;
			imagepointsPC->push_back(temp);
		}

		//vector<pcl::PointXY> 
		vector<int> inliers;
		model->setInputTarget(imagepointsPC);
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
		ransac.setDistanceThreshold(0.5);
		ransac.computeModel();
		ransac.getInliers(inliers);
		cout << inliers.size() << " inliers:\n";
		if (inliers.size() < previous_inliers){
			i = 98;
		}
		previous_inliers = inliers.size();

		inlierPoints.clear();
		inliers3D.clear();
		for (int j = 0; j < inliers.size(); j++){
			inlierPoints.push_back(nextPoints[inliers[j]]);
			inliers3D.push_back(prevPoints3D[inliers[j]]);
		}

				
	}
	

	Mat rotationArray;
	algParams.R = rotationVector.clone();
	algParams.T = translationVector.clone();
	Rodrigues(rotationVector, rotationArray);
	rotationArray = rotationArray.t();
	rotationArray.resize(4);
	rotationArray = rotationArray.t();
	translationVector.copyTo(rotationArray.col(3));


	return rotationArray;
}

Mat CReconstruction3D::estimateRotation(Mat & First, Mat & Second){
	//literal magic
	Mat H = First*Second.t();
	SVD svd(H);
	Mat r = svd.vt.t()*svd.u.t();
	if (determinant(r) < 0)
		r.row(2) *= -1.0;
	return r;
}

void CReconstruction3D::constructPointCloudInReverse(){
	algParams.nStatus = NO_ERR;

	detectSeedPointsForImage(m_mGrayImg2, m_sReconstructParams.nPoints, m_sReconstructParams.dQuality, m_sReconstructParams.dMinDistance, m_sReconstructParams.nBlSize);
	
	//calculate optical flow backwards
	vector<float> err, errTrackBack;
	vector<Point2f> prevPoints2D, seedFindBack;
	Size winSize(m_sFundamentalParams.nWinSize, m_sFundamentalParams.nWinSize);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.001);
	vector<uchar> trackBackStatus;

	calcOpticalFlowPyrLK(m_mGrayImg2, m_mGrayImg1, m_vecImgSeedPoints, prevPoints2D, m_vecOFTrackStatus, err, winSize, 4, termcrit, 0);//track points from frame 2 to frame 3
	calcOpticalFlowPyrLK(m_mGrayImg1, m_mGrayImg2, prevPoints2D, seedFindBack, trackBackStatus, errTrackBack, winSize, 4, termcrit, 0);//track points from frame 2 to frame 3

	//copy things to new locations, so the bad ones can be rejected
	m_vecImgPoints1.clear();
	m_vecImgPoints2.clear();
	for (int i = 0; i < m_vecOFTrackStatus.size(); i++){
		if (m_vecOFTrackStatus[i] && (err[i] < m_sFundamentalParams.fErr)){
			//if (ED(m_vecImgSeedPoints[i], seedFindBack[i])<1)
			{
				m_vecImgPoints1.push_back(prevPoints2D[i]);
				m_vecImgPoints2.push_back(m_vecImgSeedPoints[i]);
			}
		}
		else
			m_vecOFTrackStatus[i] = 0;
	}
	

	rejectOutlier(m_vecImgPoints2, m_vecImgPoints1, m_vImgPointsInlier2, m_vImgPointsInlier1, 0.003, 0.99);

	drawMatchingPoints(m_mCanvas, m_vImgPointsInlier1, m_vImgPointsInlier2, 0);

	//////store the points in the new seed frame
	//m_vecLastSeedFrame2D.clear();
	//for (int i = 0; i < m_vImgPointsInlier2.size(); i++)
	//	m_vecLastSeedFrame2D.push_back(m_vImgPointsInlier2[i]);

	undistortPoints(m_vImgPointsInlier1, m_vImgPointsInlier1_distort, algParams.intrinsic, algParams.distcoeff);
	undistortPoints(m_vImgPointsInlier2, m_vImgPointsInlier2_distort, algParams.intrinsic, algParams.distcoeff);

	Mat P1 = m_vecPositionMatricies[m_vecPositionMatricies.size() - 2];//this may be our issue
	Mat P2 = m_vecPositionMatricies[m_vecPositionMatricies.size() - 1];

	int cnt = 0;
	for (int i = 0; i < m_vecOFTrackStatus.size(); i++)
		if (m_vecOFTrackStatus[i])
			cnt++;

	//construct the 3D points
	point3DReconstruct(P2, P1, m_vImgPointsInlier2_distort, m_vImgPointsInlier1_distort, m_mReconstruct3D);

	vector<Point3d> vecReconstruct3D;

	cvtMat2Vec(m_mReconstruct3D, vecReconstruct3D);

	verifyNewPC(vecReconstruct3D, 2.);

	//algParams.vecRegisteredPoints3D.resize(2);
	//algParams.vecRegisteredPoints3D[0] = m_sLastPointCloud.vecPoint3D;
	//algParams.vecRegisteredPoints3D[1] = vecReconstruct3D;
	algParams.vecRegisteredPoints3D.push_back(vecReconstruct3D);

	if (algParams.nStatus == ERR_REBASE)
		return;


	//////////////////////////////////////////////////////////////////////////

	//cvtMat2Vec(m_mReconstruct3D, m_vecLastSeedFrame3D);//convert them from a matrix to a vector
	cvtMat2Vec(m_mReconstruct3D, m_sLastPointCloud.vecPoint3D);//convert them from a matrix to a vector

	
	////store the points in the new seed frame
	m_vecLastSeedFrame2D.clear();
	for (int i = 0; i < m_vImgPointsInlier2.size(); i++)
		m_vecLastSeedFrame2D.push_back(m_vImgPointsInlier2[i]);

	// Update the threshold of defocus score adaptively
	m_dDFAccum /= m_dFrameIntervalCnt;
	m_fDefocusScoreThresh = m_dDFAccum - 0.5;
	m_fDefocusScoreThresh = min(m_fDefocusScoreThresh, 7.0f);
	m_fDefocusScoreThresh = max(m_fDefocusScoreThresh, 5.5f);
	m_dDFAccum = 0;
	m_dFrameIntervalCnt = 0;

	cout << "DF Thresh: " << m_fDefocusScoreThresh << endl;
}

void CReconstruction3D::verifyNewPC( vector<Point3d>& vecReconstruct3D, double dThresh ){

	// Check the back-projection error
	vector<Point2d> vec2D2, vec2D1;
	Mat R(algParams.R.size(), CV_64F), T(algParams.R.size(), CV_64F);
	R.setTo(0); T.setTo(0);

	if (algParams.vecPointCloud.size() != 0){
		m_sLastPointCloud.mR.copyTo(R);
		m_sLastPointCloud.mT.copyTo(T);
	}

	projectPoints(vecReconstruct3D, R, T, algParams.intrinsic,
		algParams.distcoeff, vec2D1);//here is the problem, probably.

	projectPoints(vecReconstruct3D, algParams.R, algParams.T, algParams.intrinsic,
		algParams.distcoeff, vec2D2);//here is the problem, probably.

	algParams.img2.copyTo(m_mDisplay);
	for (int i = 0; i < vec2D2.size(); i++){
		circle(m_mDisplay, vec2D2[i], 2, Scalar(0, 0, 255));
		circle(m_mDisplay, m_vImgPointsInlier2[i], 3, Scalar(255, 0, 0));
	}

	imshow("Back project 2", m_mDisplay);

	algParams.img1.copyTo(m_mDisplay);
	for (int i = 0; i < vec2D1.size(); i++){
		circle(m_mDisplay, vec2D1[i], 2, Scalar(0, 0, 255));
		circle(m_mDisplay, m_vImgPointsInlier1[i], 3, Scalar(255, 0, 0));
	}
	imshow("Back project 1", m_mDisplay);

	// Compute the back-projection error
	double dErr = 0;
	for (int i = 0; i < vec2D1.size(); i++){
		dErr += ED(vec2D1[i], m_vImgPointsInlier1[i]);
		dErr += ED(vec2D2[i], m_vImgPointsInlier2[i]);
	}

	dErr /= 2 * vec2D1.size();

	//cout << "------------------------------------Back proj error: " << dErr << endl;

	if ((dErr > dThresh) && algParams.vecPointCloud.size() > 0)
		algParams.nStatus = ERR_REBASE;
}

void CReconstruction3D::ParseCommandLineArguments( int argc, char **argv )
{
	for (int i = 1; i < argc; i++)
	{
		char* pchArgument = argv[i];

		if (_strcmpi( pchArgument, "-load" ) == 0)
		{
			if (i+1<argc)
			{
				m_bLoadCloud = true;
				m_strLoadCloudName = argv[++i];
			}
			else
			{
				cout << "-load must be followed by a valid filename." << endl;
				exit( 1 );
			}
		}
		else if (_strcmpi( pchArgument, "-save" ) == 0)
		{
			if (i + 1 < argc)
			{
				m_bSaveCloud = true;
				m_strSaveCloudName = argv[++i];
			}
			else
			{
				cout << "-save must be followed by a valid filename." << endl;
				exit( 1 );
			}
		}
		else 
		{
			cout << "Unrecognized parameter: " << pchArgument << endl;
			cout << endl;
			cout << "Valid Parameters: " << endl;
			cout << argv[0] << " [-save filePathAndName] [-load filePathAndName]" << endl;
			exit( 1 );
		}
	}
}

//void CReconstruction3D::registerPointCloud2(){
//	int nLen = (int)algParams.vecPoints3D.size();
//	if (nLen <= 0) //if there is nothing to register, exit
//		return;
//
//	if (nLen == 1){//if there is only one point cloud to register, what are you doing? Put the point-cloud in the proper place for it to be visualized
//		algParams.vecRegisteredPoints3D.push_back(algParams.vecPoints3D[0]);//store the point cloud for later
//		//cout << "points2D\n";
//		for (int i = 0; i < m_vImgPointsInlier2.size(); i++){
//			m_vecSeed2D4Affine.push_back(m_vImgPointsInlier2[i]);//store the points on the screen for later
//		}
//		//m_mLastSeedFrameImgGray = m_mGrayImg2.clone();
//		return;
//	}
//
//	// Find the affine transform from the previous view pair to the current
//	vector<uchar> status;
//	vector<float> err;
//	vector<Point2f> nextPoints2D;
//	Size winSize(m_sFundamentalParams.nWinSize, m_sFundamentalParams.nWinSize);
//	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.001);
//	//algParams.lastSeedFrameImgGray = m_mGrayImg1.clone();
//	//m_vecLastSeedFrame3D = algParams.vecPoints3D[nLen - 1];//I think this is what we should be registering to?
//	//for (int i = 0; i < algParams.imgPointsInlier1.size(); i++)
//	//	m_vecLastSeedFrame2D.push_back(algParams.imgPointsInlier1[i]);
//	calcOpticalFlowPyrLK(m_mGrayImg1, m_mGrayImg2, m_vecSeed2D4Affine, nextPoints2D, status, err, winSize, 4, termcrit, 0);//track points from frame 2 to frame 3
//
//	Mat canvas;
//	// Found optical flows
//	vector<Point2f> prevPoints, nextPoints;
//	vector<Point3d>& vec3Ds = algParams.vecPoints3D[nLen - 2];
//	vector<Point3d> prevPoints3D;
//	for (int i = 0; i < nextPoints2D.size(); i++){
//		if (status[i] && (err[i] < m_sFundamentalParams.fErr)){//check to see if the optical point exists in both frames, and if it is calculated well.
//			prevPoints.push_back(m_vecSeed2D4Affine[i]);//this is the previous screenpoints
//			nextPoints.push_back(nextPoints2D[i]);//this is the current screanpoints
//			prevPoints3D.push_back(vec3Ds[i]);//okay, so this is just the 3D points we calculated on the previous frame-pair
//		}
//	}
//
//	
//	//Mat P1alt = algParams.intrinsic*rotationArray;*/
//	Mat P1alt = calculateProjectionMatrix(prevPoints3D, nextPoints);
//	//cout << "P1alt" << P1alt << endl;
//	//cout << "Calculated Projection Matrix 2" << calculateProjectionMatrix(algParams, prevPoints3D, prevPoints) << endl;
//
//	Mat r = algParams.P1;
//	Mat cameraMatrix;
//	Mat rotMatrix;
//	Mat transVect;
//	decomposeProjectionMatrix(r, cameraMatrix, rotMatrix, transVect);
//	Mat rotVect;
//	Rodrigues(rotMatrix, rotVect);
//	/*cout << "rotVect\n" << rotVect << endl;
//	cout << "cameraMatrix\n" << cameraMatrix << endl;
//	cout << "rotMatrix\n" << rotMatrix << endl;
//	cout << "transVect\n" << transVect << endl;*/
//
//	undistortPoints(prevPoints, prevPoints, algParams.intrinsic, algParams.distcoeff);
//	undistortPoints(nextPoints, nextPoints, algParams.intrinsic, algParams.distcoeff);
//
//	Mat matReconstruct3D;
//	Mat tmpImg;
//	//m_vecPositionMatricies.push_back(m_mPrevP);
//	//m_vecPositionMatricies.push_back(m_mPrevP1);
//	m_vecPositionMatricies.push_back(P1alt);
//	//nextPoints calculated purely from previous points and optical flow. D:
//	//triangulatePoints(algParams.P, algParams.P1, prevPoints, nextPoints, matReconstruct3D);//matReconstruct 3D is the triangulated points
//	//triangulatePoints(m_mPrevP1, P1alt, prevPoints, nextPoints, matReconstruct3D);//matReconstruct 3D is the triangulated points
//	//triangulatePoints(m_mPrevP1, algParams.P1, prevPoints, nextPoints, matReconstruct3D);//matReconstruct 3D is the triangulated points
//	//it takes in two points from the screen as inputs, matReconstruct3D is the return.
//	//algParams.P and algParams.P1 are the camera matrix of the first view, and camera matrix of the second view
//	//I assume these matricies are created from the reconstructFrom2Views function calling the findcameramatrix function
//	//How do we make current P the same as the previous P1? Check this. How can we then rotate the current P1 into that coordinate system?
//	
//	
//	tmpImg = matReconstruct3D.row(3).clone(); //First, take the fourth row of matReconstruct3D
//	vconcat(tmpImg, tmpImg, tmpImg); //Second, double it
//	vconcat(tmpImg, tmpImg, tmpImg); //Then double that double (four identical rows now?)
//	divide(matReconstruct3D, tmpImg, matReconstruct3D);//Okay, so we divide the matReconstruct3D by its fourth row? Unsure why/what this does.
//	
//
//	vector<Point3d> nextPoints3D;
//	vector<uchar> inliers3D;
//	cvtMat32F2Vec3D(matReconstruct3D, nextPoints3D);//okay, so this is filling the nextPoints3D vector
//	//with points that are stored in the first three rows of the matReconstruct3D matrix
//
//	Mat R, T;
//	Mat tmp;
//	algParams.vecRegisteredPoints3D.push_back(nextPoints3D);
//
//
//	//////////////////////////////////////////////////////////////////////////
//	m_vecSeed2D4Affine.clear();
//	m_vecSeed2D4Affine.resize((int)m_vImgPointsInlier2.size());
//	for (int i = 0; i < m_vImgPointsInlier2.size(); i++)
//		m_vecSeed2D4Affine[i] = m_vImgPointsInlier2[i];
//}

bool CReconstruction3D::detectSeedPoints(int nPtMax, double qualityLevel, double minDistance, int blockSize, bool bFirst){

	bool useHarrisDetector = false;
	double k = 0.04;	// Variable for Harris corner quality
	int maxCorners = nPtMax;
	/// Apply corner detection
	goodFeaturesToTrack(m_mGrayImg1, m_vecImgSeedPoints, maxCorners, qualityLevel, minDistance,
		Mat(), blockSize, useHarrisDetector, k);


	return true;
}


bool CReconstruction3D::detectSeedPointsForImage(Mat & grayImg, int nPtMax, double qualityLevel, double minDistance, int blockSize, bool bFirst){

	bool useHarrisDetector = false;
	double k = 0.04;	// Variable for Harris corner quality
	int maxCorners = nPtMax;
	/// Apply corner detection
	//goodFeaturesToTrack(grayImg, m_vecImgSeedPoints, maxCorners, qualityLevel, minDistance,
	//	Mat(), blockSize, useHarrisDetector, k);//sets m_vecImgSeedPoints

	m_pDescDetector->detect(grayImg, m_vecKeypoints2, m_mROI);
	KeyPoint::convert(m_vecKeypoints2, m_vecImgSeedPoints);

	return true;
}

void CReconstruction3D::selectKeyFrame(){
	
	if (m_vecImgPoints1.size() == 0){
		algParams.bIsKeyFrame = false;
		return;
	}

	float df_score = ComputeDFScore2(m_mGrayImg2);
	m_dDFAccum += df_score;
	m_dFrameIntervalCnt++;
	//float df_score2 = ComputeDFScore2(m_mGrayImg2);
	cout << "DF score 2:	" << df_score << endl;

	double dMean, dStd;
	// Check for the distance between views
	distanceVector2D(m_vecImgPoints1, m_vecImgPoints2, dMean, dStd);


	if ((dMean > m_dMeanThresh) && (dStd > m_dStdThresh)){
		// Check for the image quality
		//float df_score = ComputeDFScore(m_mGrayImg2);
		cout << df_score << endl;
		if (df_score > m_fDefocusScoreThresh)
			algParams.bIsKeyFrame = true;
		else
			algParams.bIsKeyFrame = false;
	}
	else
		algParams.bIsKeyFrame = false;
}

void CReconstruction3D::selectKeyFrame(vector<Point2f> & imgPoints1, vector<Point2f> & imgPoints2, vector<uchar>& mask){

	if (m_vecImgPoints1.size() == 0){
		algParams.bIsKeyFrame = false;
		return;
	}

	float df_score = ComputeDFScore2(m_mGrayImg2);
	m_dDFAccum += df_score;
	m_dFrameIntervalCnt++;
	cout << "DF score 2:	" << df_score << endl;
	//cout << "R: " << m_mPrevR << endl;
	//cout << "T: " << m_mPrevT << endl;

	// If the image is blurry, return anyway
	if (df_score < m_fDefocusScoreThresh){
		algParams.bIsKeyFrame = false;
		return;
	}

	Mat r = m_mSeedFrameR - algParams.R;
	pow(r, 2.0, r);
	reduce(r, r, 0, CV_REDUCE_SUM);
	sqrt(r, r);
	cout << "			Rotated amount:			" << r.at<double>(0) << endl;


	// If the camera made a large rotation, need a new reconstruction
	if (r.at<double>(0) > 0.08){
		algParams.bIsKeyFrame = true;
		return;
	}


	// If there left two little OF vectors for camera pose estimation, need a new reconstruction
	if (m_nSizeOfInliners < m_nMinSizeOfInliersThresh){
		algParams.bIsKeyFrame = true;
		return;
	}
	
	double dMean, dStd;
	// Check for the distance between views
	distanceVector2D(imgPoints1, imgPoints2, dMean, dStd, mask);
	

	if ((dMean > m_dMeanThresh) && (dStd > m_dStdThresh))
		algParams.bIsKeyFrame = true;
	else
		algParams.bIsKeyFrame = false;
}


void CReconstruction3D::visualizePointCloud(vector<vector<Point3d>>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, bool bStop2View){
	createPointCloudv(vecReconstruction3D, pointCloud, m_vecColors);

	//show the Point Cloud
	pcl::visualization::PCLVisualizer viewer("Reconstruction");
	
	viewer.addPointCloud(pointCloud, "Point Cloud");
	viewer.setBackgroundColor(0, 0, 0, 0);
	
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Point Cloud");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 20, "Point Cloud");
	if (bStop2View)
		while (!viewer.wasStopped())
		{
			//display the visualizer until 'q' is pressed
			viewer.spinOnce();
		}
}

void CReconstruction3D::visualizePointCloud(vector<DescriptedPC>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, bool bStop2View){
	createPointCloudv(vecReconstruction3D, pointCloud, m_vecColors);

	//show the Point Cloud
	pcl::visualization::PCLVisualizer viewer("Reconstruction");
	
	viewer.addPointCloud(pointCloud, "Point Cloud");
	viewer.setBackgroundColor(0, 0, 0, 0);

	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Point Cloud");
	
	if (bStop2View)
		while (!viewer.wasStopped())
		{
			//display the visualizer until 'q' is pressed
			viewer.spinOnce();
		}
}


void CReconstruction3D::visualizePointCloud(vector<Point3d>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, bool bStop2View){
	createPointCloud(vecReconstruction3D, pointCloud, m_vecColors);

	//show the Point Cloud
	pcl::visualization::PCLVisualizer viewer("Reconstruction");

	viewer.addPointCloud(pointCloud, "Point Cloud");
	viewer.setBackgroundColor(0, 0, 0, 0);

	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Point Cloud");

	if (bStop2View)
		while (!viewer.wasStopped())
		{
			//display the visualizer until 'q' is pressed
			viewer.spinOnce();
		}
}

void CReconstruction3D::detectCorrespondence(int nWinSize, float fErr, bool showMatchedPoints){
	/// Parameters for Shi-Tomasi algorithm
	cvtColor(algParams.img2, m_mGrayImg2, CV_RGB2GRAY);
	
	vector<Point2f> c1, c2, cc1;
	c1.clear();	c2.clear();

	for (int i = 0; i < m_vecImgSeedPoints.size(); i++)
		c1.push_back(m_vecImgSeedPoints[i]);


	vector<float> err;
	Size winSize(nWinSize, nWinSize);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.001);
	calcOpticalFlowPyrLK(m_mGrayImg1, m_mGrayImg2, c1, c2, m_vecOFTrackStatus, err, winSize, 4, termcrit, 0);
	calcOpticalFlowPyrLK(m_mGrayImg2, m_mGrayImg1, c2, cc1, m_vecOFTrackStatus, err, winSize, 4, termcrit, 0);

	m_vecImgPoints1.clear();
	m_vecImgPoints2.clear();

	for (int i = 0; i < c1.size(); i++){
		if (m_vecOFTrackStatus[i] && (err[i] < fErr) && min(c2[i].x, c2[i].y) > 0){
			//compute the distance between c1 and cc1
			//if (ED(c1[i], cc1[i]) < 1)
			{
				m_vecImgPoints1.push_back(c1[i]);
				m_vecImgPoints2.push_back(c2[i]);
			}
		}
		else
			m_vecOFTrackStatus[i] = 0;
	}

}

void CReconstruction3D::distanceVector2D(vector<Point2f>& imgPoints1, vector<Point2f>& imgPoints2, double& dMean, double& dStd, vector<uchar>& mask){
	//calculate mean distance between the pair of corresponding points
	double d = 0;
	vector<double> vDis;
	vDis.resize(imgPoints1.size());
	double dSize = 0;
	for (int i = 0; i < imgPoints1.size(); i++)
		if (mask[i]){
			vDis[i] = (double)sqrtf(pow(imgPoints1[i].x - imgPoints2[i].x, 2) + pow(imgPoints1[i].y - imgPoints2[i].y, 2));
			d += vDis[i];
			dSize++;
		}
	dMean = d / dSize;

	d = 0;
	for (int i = 0; i < imgPoints1.size(); i++){
		d += pow(vDis[i] - dMean, 2);
	}

	dStd = sqrt(d / (dSize - 1));
}

void CReconstruction3D::distanceVector2D(vector<Point2d>& imgPoints1, vector<Point2d>& imgPoints2, double& dMean, double& dStd){
	//calculate mean distance between the pair of corresponding points
	double d = 0;
	vector<double> vDis;
	vDis.resize(imgPoints1.size());
	double dSize = (double)imgPoints1.size();
	for (int i = 0; i < imgPoints1.size(); i++){
		vDis[i] = (double)sqrtf(pow(imgPoints1[i].x - imgPoints2[i].x, 2) + pow(imgPoints1[i].y - imgPoints2[i].y, 2));
		d += vDis[i];
	}
	dMean = d / dSize;

	d = 0;
	for (int i = 0; i < imgPoints1.size(); i++){
		d += pow(vDis[i] - dMean, 2);
	}

	dStd = sqrt(d / (dSize - 1));
}

void CReconstruction3D::drawMatchingPoints(Mat& canvas, vector<Point2d>& imgPoints1, vector<Point2d>& imgPoints2, int nMode){
	if (nMode == 0){
		vector<Mat> tmpimg;
		canvas = Mat::zeros(Size(algParams.img1.cols, algParams.img1.rows), CV_8UC3);
		split(algParams.img2, tmpimg);
		m_mGrayImg1.copyTo(tmpimg[0]);
		merge(tmpimg, canvas);

		for (size_t i = 0; i < imgPoints1.size(); i++){
			Point2f p1, p2;
			p1 = Point2f((float)imgPoints1[i].x, (float)imgPoints1[i].y);
			p2 = Point2f((float)imgPoints2[i].x, (float)imgPoints2[i].y);
			circle(canvas, p1, 2, Scalar(0, 0, 255));
			circle(canvas, p2, 1, Scalar(255, 255, 0));
			line(canvas, p1, p2, Scalar(255, 255, 0));
		}
	}
	else{
		canvas = Mat::zeros(Size(2 * algParams.img1.cols, algParams.img1.rows), CV_8UC3);
		algParams.img1.copyTo(canvas(Range(0, canvas.rows), Range(0, algParams.img1.cols)));
		algParams.img2.copyTo(canvas(Range(0, canvas.rows), Range(algParams.img1.cols, canvas.cols)));

		for (size_t i = 0; i < imgPoints1.size(); i++){
			Point2f p1, p2;
			float fCols = algParams.img1.cols;
			p1 = Point2f((float)imgPoints1[i].x, (float)imgPoints1[i].y);
			p2 = Point2f((float)imgPoints2[i].x +fCols, (float)imgPoints2[i].y);
			circle(canvas, p1, 2, Scalar(0, 0, 255));
			circle(canvas, p2, 1, Scalar(255, 255, 0));
			line(canvas, p1, p2, Scalar(255, 255, 0));
		}
	}
	imshow("good matched", canvas);
	cvWaitKey(1);
}

void CReconstruction3D::drawMatchingPoints(Mat& canvas, vector<Point2f>& imgPoints1, vector<Point2f>& imgPoints2, int nMode){
	if (nMode == 0){
		vector<Mat> tmpimg;
		canvas = Mat::zeros(Size(algParams.img1.cols, algParams.img1.rows), CV_8UC3);
		split(algParams.img2, tmpimg);
		m_mGrayImg1.copyTo(tmpimg[0]);
		merge(tmpimg, canvas);

		for (size_t i = 0; i < imgPoints1.size(); i++){
			circle(canvas, imgPoints1[i], 2, Scalar(0, 0, 255));
			circle(canvas, imgPoints2[i], 1, Scalar(255, 255, 0));
			line(canvas, imgPoints1[i], imgPoints2[i], Scalar(255, 255, 0));
		}
	}
	else{
		canvas = Mat::zeros(Size(2 * algParams.img1.cols, algParams.img1.rows), CV_8UC3);
		algParams.img1.copyTo(canvas(Range(0, canvas.rows), Range(0, algParams.img1.cols)));
		algParams.img2.copyTo(canvas(Range(0, canvas.rows), Range(algParams.img1.cols, canvas.cols)));

		for (size_t i = 0; i < imgPoints1.size(); i++){
			Point2f p1, p2;
			float fCols = algParams.img1.cols;
			p1 = Point2f(imgPoints1[i].x, imgPoints1[i].y);
			p2 = Point2f(imgPoints2[i].x, imgPoints2[i].y + fCols);
			circle(canvas, p1, 2, Scalar(0, 0, 255));
			circle(canvas, p2, 1, Scalar(255, 255, 0));
			line(canvas, p1, p2, Scalar(255, 255, 0));
		}
	}
	imshow("good matched", canvas);
	cvWaitKey(1);
}

void CReconstruction3D::drawMatchingPoints(Mat& img1, Mat& img2, Mat& canvas, vector<Point2f>& imgPoints1, vector<Point2f>& imgPoints2, int nMode){
	if (nMode == 0){
		vector<Mat> tmpimg;
		//canvas = img2;
		split(img2, tmpimg);
		img1.copyTo(tmpimg[0]);
		merge(tmpimg, canvas);

		for (size_t i = 0; i < imgPoints1.size(); i++){
			circle(canvas, imgPoints1[i], 2, Scalar(0, 0, 255));
			circle(canvas, imgPoints2[i], 1, Scalar(255, 255, 0));
			line(canvas, imgPoints1[i], imgPoints2[i], Scalar(255, 255, 0));
		}
	}
	else{
		canvas = Mat::zeros(Size(2 * img1.cols, img1.rows), CV_8UC3);
		img1.copyTo(canvas(Range(0, canvas.rows), Range(0, img1.cols)));
		img2.copyTo(canvas(Range(0, canvas.rows), Range(img1.cols, canvas.cols)));

		for (size_t i = 0; i < imgPoints1.size(); i++){
			Point2f p1, p2;
			float fCols = img1.cols;
			p1 = Point2f(imgPoints1[i].x, imgPoints1[i].y);
			p2 = Point2f(imgPoints2[i].x, imgPoints2[i].y + fCols);
			circle(canvas, p1, 2, Scalar(0, 0, 255));
			circle(canvas, p2, 1, Scalar(255, 255, 0));
			line(canvas, p1, p2, Scalar(255, 255, 0));
		}
	}
	imshow("good matched", canvas);
	cvWaitKey(1);
}

void CReconstruction3D::drawMatchingPoints(Mat& canvas, vector<Point2d>& imgPoints1, vector<Point2f>& imgPoints2, int nMode){
	if (nMode == 0){
		vector<Mat> tmpimg;
		canvas = Mat::zeros(Size(algParams.img1.cols, algParams.img1.rows), CV_8UC3);
		split(algParams.img2, tmpimg);
		m_mGrayImg1.copyTo(tmpimg[0]);
		merge(tmpimg, canvas);

		for (size_t i = 0; i < imgPoints1.size(); i++){
			circle(canvas, imgPoints1[i], 2, Scalar(0, 0, 255));
			circle(canvas, imgPoints2[i], 1, Scalar(255, 255, 0));
			line(canvas, imgPoints1[i], imgPoints2[i], Scalar(255, 255, 0));
		}
	}
	else{
		canvas = Mat::zeros(Size(2 * algParams.img1.cols, algParams.img1.rows), CV_8UC3);
		algParams.img1.copyTo(canvas(Range(0, canvas.rows), Range(0, algParams.img1.cols)));
		algParams.img2.copyTo(canvas(Range(0, canvas.rows), Range(algParams.img1.cols, canvas.cols)));

		for (size_t i = 0; i < imgPoints1.size(); i++){
			Point2f p1, p2;
			float fCols = algParams.img1.cols;
			p1 = Point2f(imgPoints1[i].x, imgPoints1[i].y);
			p2 = Point2f(imgPoints2[i].x, imgPoints2[i].y + fCols);
			circle(canvas, p1, 2, Scalar(0, 0, 255));
			circle(canvas, p2, 1, Scalar(255, 255, 0));
			line(canvas, p1, p2, Scalar(255, 255, 0));
		}
	}
	imshow("good matched", canvas);
	cvWaitKey(1);
}


void CReconstruction3D::rejectOutlier(const vector<Point2d>& imagePoints1, const vector<Point2f>& imagePoints2, vector<uchar>& imagePointsInlier, 
	float minDist, float qualityRatio, bool bUpdateOFStt){
	//Mat F, status;

	findFundamentalMat(imagePoints1, imagePoints2, CV_FM_LMEDS, minDist, qualityRatio, imagePointsInlier);

}

void CReconstruction3D::rejectOutlier(const vector<Point2d>& imagePoints1, const vector<Point2d>& imagePoints2, vector<Point2d>& imagePointsInlier1, 
	vector<Point2d>& imagePointsInlier2, float minDist, float qualityRatio, bool bUpdateOFStt){
	Mat F;
	imagePointsInlier1.clear();
	imagePointsInlier2.clear();

	findFundamentalMat(imagePoints1, imagePoints2, CV_FM_LMEDS, minDist, qualityRatio, m_mInlierMask);
	//findFundamentalMat(imagePoints1, imagePoints2, CV_FM_LMEDS, minDist, qualityRatio, m_vecInlinerStatus);

	int nLen = (int)sum(m_mInlierMask).val[0];

	imagePointsInlier1.resize(nLen);
	imagePointsInlier2.resize(nLen);
	for (int i = 0, k = 0; i < m_mInlierMask.rows; i++){
		int checkInlier = m_mInlierMask.at<uchar>(i, 0);
		if (checkInlier == 1){
			imagePointsInlier1[k] = imagePoints1[i];
			imagePointsInlier2[k] = imagePoints2[i];
			k++;
		}
	}

	// Update the OF tracking status
	if (bUpdateOFStt)
		for (int i = 0, k = 0; i < m_vecOFTrackStatus.size(); i++)
			if (m_vecOFTrackStatus[i])
				m_vecOFTrackStatus[i] = m_mInlierMask.at<uchar>(k++, 0);
}

void CReconstruction3D::rejectOutlier(const vector<Point2f>& imagePoints1, const vector<Point2f>& imagePoints2, vector<Point2f>& imagePointsInlier1,
	vector<Point2f>& imagePointsInlier2, float minDist, float qualityRatio, bool bUpdateOFStt){
	
	findFundamentalMat(imagePoints1, imagePoints2, CV_FM_LMEDS, minDist, qualityRatio, m_mInlierMask);

	imagePointsInlier1.clear();
	imagePointsInlier2.clear();

	int nLen = (int)sum(m_mInlierMask).val[0];

	imagePointsInlier1.resize(nLen);
	imagePointsInlier2.resize(nLen);
	for (int i = 0, k = 0; i < m_mInlierMask.rows; i++){
		int checkInlier = m_mInlierMask.at<uchar>(i, 0);
		if (checkInlier == 1){
			imagePointsInlier1[k] = imagePoints1[i];
			imagePointsInlier2[k] = imagePoints2[i];
			k++;
		}
	}

	// Update the OF tracking status
	if (bUpdateOFStt)
		for (int i = 0, k = 0; i < m_vecOFTrackStatus.size(); i++)
			if (m_vecOFTrackStatus[i])
				m_vecOFTrackStatus[i] = m_mInlierMask.at<uchar>(k++, 0);
}

void CReconstruction3D::rejectOutlier(const vector<Point2f>& imagePoints1, const vector<Point2f>& imagePoints2, vector<uchar>& imagePointsInlier, 
	float minDist, float qualityRatio, bool bUpdateOFStt){
	Mat F, status;

	findFundamentalMat(imagePoints1, imagePoints2, CV_FM_LMEDS, minDist, qualityRatio, imagePointsInlier);

	// Update the OF tracking status
	if (bUpdateOFStt)
		for (int i = 0, k = 0; i < m_vecOFTrackStatus.size(); i++)
			if (m_vecOFTrackStatus[i])
				m_vecOFTrackStatus[i] = imagePointsInlier[k++];
}

void CReconstruction3D::cvtMat2Vec(Mat& input, vector<Point2d>& output){
	output.clear();
	Point2d point;
	for (int i = 0; i < input.rows; i++){
		point.x = input.at<double>(i, 0);
		point.y = input.at<double>(i, 1);
		output.push_back(point);
	}

}

void CReconstruction3D::cvtMat2Vec(Mat& input, vector<Point3d>& output){
	output.clear();
	Point3d point;
	int nSize = input.cols;
	output.resize(nSize);
	for (int i = 0; i < input.cols; i++){
		point.x = input.at<double>(0, i);
		point.y = input.at<double>(1, i);
		point.z = input.at<double>(2, i);
		output[i] = point;
	}
}

void CReconstruction3D::cvtMat2Vec(Mat& input, vector<Point3f>& output){
	output.clear();
	Point3d point;
	int nSize = input.cols;
	output.resize(nSize);
	for (int i = 0; i < input.cols; i++){
		point.x = input.at<double>(0, i);
		point.y = input.at<double>(1, i);
		point.z = input.at<double>(2, i);
		output[i] = point;
	}
}

void CReconstruction3D::cvtMat32F2Vec3D(Mat& input, vector<Point3d>& output){
	output.clear();
	Point3d point;
	int nSize = input.cols;
	output.resize(nSize);
	for (int i = 0; i < input.cols; i++){
		point.x = input.at<float>(0, i);
		point.y = input.at<float>(1, i);
		point.z = input.at<float>(2, i);
		output[i] = point;
	}
}

void CReconstruction3D::cvtVec2Mat(vector<Point3d>& vinput, Mat& moutput){
	moutput = Mat(Size(vinput.size(), 3), CV_64FC1);

	for (int i = 0; i < vinput.size(); i++){
		double arr[3];
		arr[0] = vinput[i].x;
		arr[1] = vinput[i].y;
		arr[2] = vinput[i].z;
		Mat tmp(Size(1, 3), CV_64FC1, arr);
		tmp.copyTo(moutput.col(i));
	}
}


void CReconstruction3D::cvtVec2Mat(vector<Point2d>& vinput, Mat& moutput){
	moutput = Mat(Size(vinput.size(), 2), CV_64FC1);

	for (int i = 0; i < vinput.size(); i++){
		double arr[2];
		arr[0] = vinput[i].x;
		arr[1] = vinput[i].y;
		Mat tmp(Size(1, 3), CV_64FC1, arr);
		tmp.copyTo(moutput.col(i));
	}
}

void CReconstruction3D::cvtVec2Mat(vector<Point3f>& vinput, Mat& moutput){
	moutput = Mat(Size(vinput.size(), 3), CV_32FC1);

	for (int i = 0; i < vinput.size(); i++){
		double arr[3];
		arr[0] = vinput[i].x;
		arr[1] = vinput[i].y;
		arr[2] = vinput[i].z;
		Mat tmp(Size(1, 3), CV_32FC1, arr);
		tmp.copyTo(moutput.col(i));
	}
}

void CReconstruction3D::mySVD(Mat& E, Mat& U, Mat& S, Mat& V){
	//calculate U by using Gram-Schmidt
	//calculate eigenvectors and eigenvalues of A*A.t()
	Mat A, eigenvalues, eigenvectors;
	A = E*E.t();
	eigen(A, eigenvalues, eigenvectors);
	/*cout << A << endl;
	cout << eigenvalues << endl;
	cout << eigenvectors << endl;*/

	Mat u1, u2, u3;
	u1 = eigenvectors.col(0).clone();
	u1 = u1 / norm(u1);
	u2 = eigenvectors.col(1) - u1.dot(eigenvectors.col(1))*u1;
	u2 = u2 / norm(u2);
	u3 = eigenvectors.col(2) - u1.dot(eigenvectors.col(2))*u1 - u2.dot(eigenvectors.col(2))*u2;
	u3 = u3 / norm(u3);
	U = Mat(Size(3, 3), CV_64FC1);
	u1.copyTo(U.col(0));
	u2.copyTo(U.col(1));
	u3.copyTo(U.col(2));
	U = (-1)*U.t();

	A = E.t()*E;
	eigen(A, eigenvalues, eigenvectors);
	u1 = eigenvectors.col(0).clone();
	u1 = u1 / norm(u1);
	u2 = eigenvectors.col(1) - u1.dot(eigenvectors.col(1))*u1;
	u2 = u2 / norm(u2);
	u3 = eigenvectors.col(2) - u1.dot(eigenvectors.col(2))*u1 - u2.dot(eigenvectors.col(2))*u2;
	u3 = u3 / norm(u3);
	V = Mat(Size(3, 3), CV_64FC1);
	u1.copyTo(V.col(0));
	u2.copyTo(V.col(1));
	u3.copyTo(V.col(2));
	V = (-1)*V.t();
	V.col(1) = (-1)*V.col(1);

	double tmp = eigenvalues.at<double>(2, 0);
	sqrt(eigenvalues, eigenvalues);
	eigenvalues.at<double>(2, 0) = tmp;


	Mat D = Mat::zeros(Size(3, 3), CV_64FC1);
	D.at<double>(0, 0) = eigenvalues.at<double>(0, 0);
	D.at<double>(1, 1) = eigenvalues.at<double>(1, 0);
	D.at<double>(2, 2) = eigenvalues.at<double>(2, 0);
	S = D.clone();

}

Mat CReconstruction3D::camMatrix(Mat& R, Mat& T){
	Mat P = Mat::zeros(3, 4, CV_64FC1);
	R.copyTo(P.colRange(0, 3));
	T.copyTo(P.col(3));
	return P;
}

Mat CReconstruction3D::camMatrix(Mat& R, Mat& T, Mat& intrinsic){
	Mat P = Mat::zeros(3, 4, CV_64FC1);
	R.copyTo(P.colRange(0, 3));
	T.copyTo(P.col(3));
	P = intrinsic*P;

	return P;
}

Mat CReconstruction3D::triangulateMidPoint(vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2, Mat P, Mat P1){
	Mat point3D;
	Mat M1, M2;
	point3D = Mat::zeros(imagePoints1.size(), 3, CV_64FC1);
	P = P.t();
	P1 = P1.t();
	M1 = P.colRange(0, 3).rowRange(0, 3);
	M2 = P1.colRange(0, 3).rowRange(0, 3);

	Mat c1 = -M1.inv()*P.col(3);
	Mat c2 = -M2.inv()*P1.col(3);
	for (unsigned i = 0; i < imagePoints1.size(); i++){
		double u1tmp[3] = { imagePoints1[i].x, imagePoints1[i].y, 1 };
		double u2tmp[3] = { imagePoint2[i].x, imagePoint2[i].y, 1 };
		Mat u, A;
		u = Mat(3, 1, CV_64FC1, u1tmp);
		Mat a1 = M1.inv()*u;
		u = Mat(3, 1, CV_64FC1, u2tmp);
		Mat a2 = M2.inv()*u;
		hconcat(a1, -a2, A);
		Mat y = c2 - c1;
		u = A.t()*A;
		Mat alpha = u.inv()*A.t()*y;
		Mat p = (c1 + alpha.at<double>(0, 0) *a1 + c2 + alpha.at<double>(1, 0)*a2) / 2;
		p = p.t();
		p.copyTo(point3D.row(i));
	}
	return point3D;
}

void CReconstruction3D::chooseRealizableSolution(vector<Mat>& Rs, vector<Mat>& Ts, vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2, Mat& intrinsic, Mat& R, Mat& T){
	vector<int> numNegative;
	Mat tmp, tmp1, P, P1;
	Mat m1, m2;
	tmp = Mat::eye(3, 3, CV_64FC1);
	tmp1 = Mat::zeros(3, 1, CV_64FC1);
	P = camMatrix(tmp, tmp1, intrinsic);
	P = P.t();

	for (unsigned i = 0; i < Ts.size(); i++){
		//try each set of rotations and translations
		tmp = Rs[i].clone();
		tmp1 = Ts[i].clone();
		P1 = camMatrix(tmp, tmp1, intrinsic);//make a projection matrix out of it.
		P1 = P1.t();
		//triangulate mid points
		m1 = triangulateMidPoint(imagePoints1, imagePoint2, P, P1);
		m2 = m1.clone();
		for (int k = 0; k < m1.rows; k++)
			m2.row(k) = m2.row(k) - Ts[i].t();
		m2 = m2 * Rs[i].t();

		bitwise_or(m1.col(2) < 0, m2.col(2)<0, tmp);
		numNegative.push_back((int)sum(tmp).val[0] / 255);
	}

	int nMinIdx = 0;
	int nMin = (int)m1.rows;
	for (int i = 0; i < numNegative.size(); i++)
		if (numNegative[i] < nMin)
			nMinIdx = i;

	R = Rs[nMinIdx].clone().t();
	T = Ts[nMinIdx].clone();
}

void CReconstruction3D::findCameraMatrix(vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2,
	Mat& intrinsic, Mat& distcoeff, Mat& P, Mat& P1, Mat& F, int& nStatus){
	Mat E;
	Mat& R = algParams.R; 
	Mat& T = algParams.T;
	
	F = findFundamentalMat(imagePoints1, imagePoint2, CV_FM_8POINT);
	//F = findFundamentalMat(imagePoints1, imagePoints2, CV_FM_8POINT);
	//infer the essential matrix from fundamental matrix
	E = intrinsic.t()*F*intrinsic;



	//calculate 4 possible values of R and t
	Mat S, U, V, D;
	D = Mat::zeros(Size(3, 3), CV_64FC1);
	//cvSVD(&CvMat(E), &CvMat(S), &CvMat(U), &CvMat(V));
	SVD::compute(E, S, U, V);
	double e = (S.at<double>(0, 0) + S.at<double>(1, 0)) / 2;
	D.at<double>(0, 0) = e;
	D.at<double>(1, 1) = e;
	D.at<double>(2, 2) = 0;

	E = U*D*V;
	SVD::compute(E, S, U, V);
	//mySVD(E, U, S, V);	
	double arr[9] = { 0, -1, 0, 1, 0, 0, 0, 0, 1 };
	double arr1[9] = { 0, 1, 0, -1, 0, 0, 0, 0, 0 };
	Mat W = Mat(Size(3, 3), CV_64FC1, arr);
	Mat Z = Mat(Size(3, 3), CV_64FC1, arr1);
	Mat R1 = U*W*V;
	Mat R2 = U*W.t()*V;
	R1 = determinant(R1) < 0 ? -R1 : R1;
	R2 = determinant(R2) < 0 ? -R2 : R2;
	Mat Tx = U*Z*U.t();
	Mat t = Mat::zeros(Size(1, 3), CV_64FC1);
	t.at<double>(0, 0) = Tx.at<double>(2, 1);
	t.at<double>(1, 0) = Tx.at<double>(0, 2);
	t.at<double>(2, 0) = Tx.at<double>(1, 0);
	vector<Mat> Rs, Ts;
	Rs.push_back(R2); Rs.push_back(R2);
	Rs.push_back(R1); Rs.push_back(R1);
	Ts.push_back(t); Ts.push_back(-t);
	Ts.push_back(t); Ts.push_back(-t);


	chooseRealizableSolution(Rs, Ts, imagePoints1, imagePoint2, intrinsic, R, T);


	P1 = Mat(3, 4, CV_64FC1);
	P = Mat::zeros(3, 4, CV_64FC1);
	//calculate P matrix (I|0)
	Mat tmp = Mat::eye(3, 3, CV_64FC1);
	tmp.copyTo(P.colRange(0, 3));

	//calculate P1 matrix (I|0)
	R = R.t();
	T = -T.t()*R;
	T = T.t();
	R.copyTo(P1.colRange(0, 3));
	T.copyTo(P1.col(3));

}

void CReconstruction3D::reconstructFrom2Views(){

	Mat matReconstruct3D;
	vector<Point3d> vecReconstruction3D;

	// Estimate the fundamental matrix
	estimateFundamentalMatrix();


	if (algParams.nStatus == ERR_FIND_CAM_MATRIX){
		return;
	}

	reconstruct3DPoints(vecReconstruction3D);

	
	// Save the camera matrices
	if (m_vecCameraMatrices.size() == 0){
		m_vecCameraMatrices.push_back(algParams.P);
		m_vecCameraMatrices.push_back(algParams.P1);
	}
	else
		m_vecCameraMatrices.push_back(algParams.P1);

	algParams.vecPointCloud.push_back(m_sLastPointCloud);
}

bool CReconstruction3D::detectSeedPoints4PCloud(int nPtMax, double qualityLevel, double minDistance, int blockSize, bool bFirst){
	///// Parameters for Shi-Tomasi algorithm
	//cvtColor(algParams.img1, m_mGrayImg1, CV_RGB2GRAY);
	//float df_score = ComputeDFScore(m_mGrayImg1);	// Defocus score
	//cout << df_score << endl;
	//if (bFirst)
	//	if (df_score > m_mFDefocusScoreThresh)
	//		return false;

	//bool useHarrisDetector = false;
	//double k = 0.04;	// Variable for Harris corner quality
	//int maxCorners = nPtMax;
	///// Apply corner detection
	//if ((int)algParams.vecPoints3D.size() == 0)
	//	goodFeaturesToTrack(m_mGrayImg1, m_vecImgSeedPoints, maxCorners, qualityLevel, minDistance,
	//	Mat(), blockSize, useHarrisDetector, k);
	//else
	//{
	//	Mat img = algParams.img1.clone();
	//	for (int i = 0; i < m_vProjectedPts2DWhole.size(); i++){
	//		circle(img, m_vProjectedPts2DWhole[i], 8, Scalar(0, 255, 0));
	//	}

	//	m_mCornerMask.create(algParams.img1.size(), CV_8UC1);
	//	m_mCornerMask.setTo(0);
	//	int nCorners = (int)algParams.imgPointsInlier2.size();

	//	for (int i = 0; i < nCorners; i++){
	//		m_mCornerMask.at<uchar>((int)algParams.imgPointsInlier2[i].y, (int)algParams.imgPointsInlier2[i].x) = 255;
	//		circle(img, algParams.imgPointsInlier2[i], 1, Scalar(0, 0, 255));
	//		//circle(img, m_vProjectedPts2DWhole[i], 8, Scalar(0, 255, 0));
	//	}

	//	Mat strel = getStructuringElement(CV_SHAPE_ELLIPSE, Size((int)2 * m_mFundamentalParams.dMinDistance, (int)2 * m_mFundamentalParams.dMinDistance));
	//	dilate(m_mCornerMask, m_mCornerMask, strel);
	//	bitwise_not(m_mCornerMask, m_mCornerMask);
	//	//imshow("Mask", m_mCornerMask);
	//	//imshow("corner", img);
	//	//cvWaitKey(0);
	//	goodFeaturesToTrack(m_mGrayImg1, m_vecImgSeedPoints, maxCorners - nCorners, qualityLevel, minDistance,
	//		m_mCornerMask, blockSize, useHarrisDetector, k);

	//	// Insert the corners from previous reconstruction to the new corner set
	//	m_vecImgSeedPoints.insert(m_vecImgSeedPoints.begin(), algParams.imgPointsInlier2.begin(), algParams.imgPointsInlier2.end());

	//}

	return true;
}


void CReconstruction3D::reconstruct3DPoints(vector<Point3d>& vecReconstruction3D){

	//detectSeedPoints4PCloud(algParams, m_sReconstructParams.nPoints, m_sReconstructParams.dQuality, m_sReconstructParams.dMinDistance, m_sReconstructParams.nBlSize);
	detectSeedPoints(m_sReconstructParams.nPoints, m_sReconstructParams.dQuality, m_sReconstructParams.dMinDistance, m_sReconstructParams.nBlSize);

	detectCorrespondence(m_sReconstructParams.nWinSize, m_sReconstructParams.fErr, false);

	rejectOutlier(m_vecImgPoints1, m_vecImgPoints2, m_vImgPointsInlier1, m_vImgPointsInlier2, 0.01, 0.99);

	drawMatchingPoints(m_mCanvas, m_vImgPointsInlier1, m_vImgPointsInlier2);

	//// Remove the points being outside the image
	//int nLen = (int)m_vImgPointsInlier2.size();
	//int nXLimit = algParams.img1.cols;
	//int nYLimit = algParams.img1.rows;
	//for (int i = 0; i < nLen; i++){
	//	Point2d pt = m_vImgPointsInlier2[i];
	//	if ((pt.x < 0) || (pt.y < 0) || (pt.x >= nXLimit) || (pt.y >= nYLimit))
	//		m_vecTrackStatus[i] = 0;
	//}

	point3DReconstruct1(algParams.P, algParams.P1, algParams.intrinsic, algParams.distcoeff, m_vImgPointsInlier1, m_vImgPointsInlier2, vecReconstruction3D);
}

void CReconstruction3D::point3DReconstruct1(Mat& P, Mat& P1, Mat &intrinsic, Mat &distcoeff, vector<Point2d> vimgPointsInlier1, vector<Point2d> vimgPointsInlier2, vector<Point3d> &vecReconstruction3D){

	undistortPoints(vimgPointsInlier1, m_vImgPointsInlier1_distort, intrinsic, distcoeff);
	undistortPoints(vimgPointsInlier2, m_vImgPointsInlier2_distort, intrinsic, distcoeff);

	point3DReconstruct(P, P1, m_vImgPointsInlier1_distort, m_vImgPointsInlier2_distort, m_mReconstruct3D);
	cvtMat2Vec(m_mReconstruct3D, vecReconstruction3D);
}

void CReconstruction3D::findCameraMatrixIter(vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2, Mat& intrinsic, Mat& distcoeff, Mat& P, Mat& P1, Mat& F, int& nStatus){

	//find the camera matrices from a set of corresponding points
	findCameraMatrix(imagePoints1, imagePoint2, intrinsic, distcoeff, P, P1, F, nStatus);


	//try to reconstruct 3D point object
	//attempt to calculate the 3D point cloud from the set of corresponding points
	//in the fundamental matrix calculation
	//vector<Point3d> vecReconstruction3D;
	double avgErr;
	vector<double> verr;
	Mat tmpR, tmpT;
	Mat& R = algParams.R;
	Mat& T = algParams.T;

	//get the rotation and translation from P1 camera matrix (in the relation to the fixed camera)
	P1.colRange(0, 3).copyTo(R);
	P1.col(3).copyTo(T);
	//get the rotation and translation matrices in the relation to the second image	
	tmpR = R.t();
	tmpT = -tmpR*T;

	//reconstruct 3D points from pairs of corresponding points
	point3DReconstruct1(P, P1, intrinsic, distcoeff, imagePoints1, imagePoint2, m_sLastPointCloud.vecPoint3D);

	//calculate the reprojection error in the fundamental matrix calculation
	reprojectionErr(m_sLastPointCloud.vecPoint3D, tmpR, tmpT, intrinsic, distcoeff, imagePoint2, avgErr, verr);

	//update the corresponding points (estimate again the camera matrix)
	updateCorrespondingPoint(P, P1, intrinsic, distcoeff, imagePoints1, imagePoint2, m_d3DReprojectErrThresh, avgErr, verr);

	// Convert from matrix form to vector form. Needed for back-project in tracking
	Rodrigues(algParams.R, algParams.R);


	double dMax = 0;
	for (int i = 0; i < 3; i++)
		if (dMax < abs(R.at<double>(i, 0)))
			dMax = abs(R.at<double>(i, 0));
	cout << "Max R:	" << dMax << endl;

	if (dMax > m_dRotationMaxThresh)
		nStatus = ERR_FIND_CAM_MATRIX;
}

//update the right corresponding points
void CReconstruction3D::updateCorrespondingPoint(Mat& P, Mat& P1, Mat &intrinsic, Mat &distcoeff, vector<Point2d>& vimgPointsInlier1, vector<Point2d>& vimgPointsInlier2, double thres, double &avgErr, vector<double> &verr){

	Mat tmpR, tmpT;
	Mat& R = algParams.R; 
	Mat& T = algParams.T;

	//vector<Point3d> vecReconstruction3D;
	Mat F;
	int nStatus = 0;

	if (avgErr < thres){
		return;
	}
	else{
		for (int i = 0; i < 10; i++){
			//process the iteration until the reprojection error is smaller than threshold
			auto biggest = std::max_element(std::begin(verr), std::end(verr));
			cout << "Max element is " << *biggest << " at position " << distance(begin(verr), biggest) << endl;
			//delete the points at the position and calculate again the camera matrix
			vimgPointsInlier1.erase(vimgPointsInlier1.begin() + distance(begin(verr), biggest));
			vimgPointsInlier2.erase(vimgPointsInlier2.begin() + distance(begin(verr), biggest));

			//find the camera matrices from a set of corresponding points
			findCameraMatrix(vimgPointsInlier1, vimgPointsInlier2, intrinsic, distcoeff, P, P1, F, nStatus);

			//get the rotation and translation from P1 camera matrix (in the relation to the fixed camera)
			P1.colRange(0, 3).copyTo(R);
			P1.col(3).copyTo(T);
			//get the rotation and translation matrices in the relation to the second image	
			tmpR = R.t();
			tmpT = -tmpR*T;

			//reconstruct 3D points from pairs of corresponding points
			point3DReconstruct1(P, P1, intrinsic, distcoeff, vimgPointsInlier1, vimgPointsInlier2, m_sLastPointCloud.vecPoint3D);

			//calculate the reprojection error in the fundamental matrix calculation
			reprojectionErr(m_sLastPointCloud.vecPoint3D, tmpR, tmpT, intrinsic, distcoeff, vimgPointsInlier2, avgErr, verr);
			//out of the loop if the average error is smaller than threshold
			if (avgErr < thres)
				break;
		} // end of for
	} //end of if
}


void CReconstruction3D::reprojectionErr(vector<Point3d> &vecReconstruction3D, Mat& R, Mat& T, Mat &intrinsic, Mat &distcoeff, vector<Point2d> &imgPts, double &avgErr, vector<double> &verr){

	//check id the 3D cloud point vector and image point vector are valid
	if (vecReconstruction3D.size() == 0 || imgPts.size() == 0)
		return;

	vector<Point2d> tmpPts;
	//project back the reconstructed points from 3D to 2D
	projectPoints(vecReconstruction3D, R, T, intrinsic, distcoeff, tmpPts);

	avgErr = 0;
	verr.resize(imgPts.size());

	for (int i = 0; i < tmpPts.size(); i++){
		verr[i] = ED(tmpPts[i], imgPts[i]);
		avgErr += verr[i];
	}

	avgErr /= tmpPts.size();

	/*if (err > 70)
	algParams.nStatus = ERR_FIND_CAM_MATRIX;*/
	//check if the reprojection error is larger than the threshold
	//go through the corresponding points to delete some points
	//calculate the 3D point cloud again

	cout << "Average projection error " << avgErr << endl;
}


void CReconstruction3D::estimateFundamentalMatrix(){

	detectSeedPoints(m_sFundamentalParams.nPoints, m_sFundamentalParams.dQuality, m_sFundamentalParams.dMinDistance, m_sFundamentalParams.nBlSize);

	detectCorrespondence(m_sFundamentalParams.nWinSize, m_sFundamentalParams.fErr, false);

	//drawMatchingPoints(m_mCanvas, m_vecImgPoints1, m_vecImgPoints2);

	rejectOutlier(m_vecImgPoints1, m_vecImgPoints2, m_vImgPointsInlier1, m_vImgPointsInlier2, 0.001, 0.99);

	drawMatchingPoints(m_mCanvas, m_vImgPointsInlier1, m_vImgPointsInlier2);

	findCameraMatrixIter(m_vImgPointsInlier1, m_vImgPointsInlier2, algParams.intrinsic, algParams.distcoeff, algParams.P,
		algParams.P1, m_mF, algParams.nStatus);

}




void CReconstruction3D::createPointCloud(vector<Point3d>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZ>::Ptr &pointCloud){
	//create a Point Cloud	
	pointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	pointCloud->points.resize(vecReconstruction3D.size());
	for (size_t i = 0; i < vecReconstruction3D.size(); i++){
		Point3f p3f((float)vecReconstruction3D[i].x, (float)vecReconstruction3D[i].y, (float)vecReconstruction3D[i].z);
		pointCloud->points[i].x = p3f.x;
		pointCloud->points[i].y = p3f.y;
		pointCloud->points[i].z = -p3f.z;
	}
}

void CReconstruction3D::createPointCloud(vector<vector<Point3d>>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZ>::Ptr &pointCloud){
	//create a Point Cloud	
	pointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
	int nSize = 0;
	for (size_t i = 0; i < vecReconstruction3D.size(); i++)
		for (int j = 0; j < vecReconstruction3D[i].size(); j++)
			nSize++;
	pointCloud->points.resize(nSize);

	for (size_t i = 0; i < vecReconstruction3D.size(); i++){
		for (int j = 0; j < vecReconstruction3D[i].size(); j++){
			Point3f p3f((float)vecReconstruction3D[i][j].x, (float)vecReconstruction3D[i][j].y, (float)vecReconstruction3D[i][j].z);
			pointCloud->points[i].x = p3f.x;
			pointCloud->points[i].y = p3f.y;
			pointCloud->points[i].z = -p3f.z;
		}
	}
}

void CReconstruction3D::createPointCloud(vector<Point3d>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, vector<Scalar> &colors){
	//create a Point Cloud	
	pointCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	int num = 0;

	//pointCloud->points.resize((int)vecReconstruction3D.size());
	int idx = 0;
	Scalar color;
	color = colors[0];

	uint32_t rgb = (static_cast<uint32_t>(color.val[0]) | static_cast<uint32_t>(color.val[1]) | static_cast<uint32_t>(color.val[2]));
	for (size_t j = 0; j < vecReconstruction3D.size(); j++){


		pcl::PointXYZRGB pointxyzrgb;
		Point3f p3f((float)vecReconstruction3D[j].x, (float)vecReconstruction3D[j].y, (float)vecReconstruction3D[j].z);
		pointxyzrgb.x = p3f.x;
		pointxyzrgb.y = p3f.y;
		pointxyzrgb.z = -p3f.z;
		//pointxyzrgb.rgb = *reinterpret_cast<float*>(&rgb);
		pointxyzrgb.r = color.val[0];
		pointxyzrgb.g = color.val[1];
		pointxyzrgb.b = color.val[2];

		pointCloud->push_back(pointxyzrgb);
	}
}

void CReconstruction3D::createPointCloudv(vector<vector<Point3d>>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, vector<Scalar> &colors){

	//create a Point Cloud	
	pointCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	int num = 0;
	for (size_t i = 0; i < vecReconstruction3D.size(); i++){
		num += vecReconstruction3D[i].size();
	}
	pointCloud->points.resize(num);
	int idx = 0;
	Scalar color;
	for (size_t j = 0; j < vecReconstruction3D.size(); j++){
		if (j < 10)
			color = colors[j];
		else{
			color.val[0] = rand() % 255;
			color.val[1] = rand() % 255;
			color.val[2] = rand() % 255;
		}

		uint32_t rgb = (static_cast<uint32_t>(color.val[0]) | static_cast<uint32_t>(color.val[1]) | static_cast<uint32_t>(color.val[2]));

		for (size_t i = 0; i < vecReconstruction3D[j].size(); i++){
			pcl::PointXYZRGB pointxyzrgb;
			Point3f p3f((float)vecReconstruction3D[j][i].x, (float)vecReconstruction3D[j][i].y, (float)vecReconstruction3D[j][i].z);
			pointxyzrgb.x = p3f.x;
			pointxyzrgb.y = p3f.y;
			pointxyzrgb.z = -p3f.z;
			//pointxyzrgb.rgb = *reinterpret_cast<float*>(&rgb);
			pointxyzrgb.r = color.val[0];
			pointxyzrgb.g = color.val[1];
			pointxyzrgb.b = color.val[2];

			pointCloud->push_back(pointxyzrgb);
		}
	}
}

void CReconstruction3D::createPointCloudv(vector<DescriptedPC>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, vector<Scalar> &colors){

	//create a Point Cloud	
	pointCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	int num = 0;
	for (size_t i = 0; i < vecReconstruction3D.size(); i++){
		num += vecReconstruction3D[i].vecPoint3D.size();
	}
	pointCloud->points.resize(num);
	int idx = 0;
	Scalar color;
	for (size_t j = 0; j < vecReconstruction3D.size(); j++){
		if (j < 10)
			color = colors[j];
		else{
			color.val[0] = rand() % 255;
			color.val[1] = rand() % 255;
			color.val[2] = rand() % 255;
		}

		uint32_t rgb = (static_cast<uint32_t>(color.val[0]) | static_cast<uint32_t>(color.val[1]) | static_cast<uint32_t>(color.val[2]));

		for (size_t i = 0; i < vecReconstruction3D[j].vecPoint3D.size(); i++){
			pcl::PointXYZRGB pointxyzrgb;
			Point3f p3f((float)vecReconstruction3D[j].vecPoint3D[i].x, (float)vecReconstruction3D[j].vecPoint3D[i].y, 
				(float)vecReconstruction3D[j].vecPoint3D[i].z);
			pointxyzrgb.x = p3f.x;
			pointxyzrgb.y = p3f.y;
			pointxyzrgb.z = -p3f.z;
			//pointxyzrgb.rgb = *reinterpret_cast<float*>(&rgb);
			pointxyzrgb.r = color.val[0];
			pointxyzrgb.g = color.val[1];
			pointxyzrgb.b = color.val[2];

			pointCloud->push_back(pointxyzrgb);
		}
	}
}


void CReconstruction3D::createColors(vector<Scalar>& colors){
	colors.clear();
	colors.push_back(Scalar(255, 0, 0));
	colors.push_back(Scalar(127, 127, 255));
	colors.push_back(Scalar(0, 255, 0));
	colors.push_back(Scalar(255, 255, 0));
	colors.push_back(Scalar(0, 255, 255));
	colors.push_back(Scalar(255, 0, 255));
	colors.push_back(Scalar(127, 0, 0));
	colors.push_back(Scalar(0, 127, 0));
	colors.push_back(Scalar(0, 0, 127));
	colors.push_back(Scalar(127, 127, 0));
	colors.push_back(Scalar(0, 127, 127));
	colors.push_back(Scalar(127, 0, 127));
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> CReconstruction3D::normalsVis(
	pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
	// --------------------------------------------------------
	// -----Open 3D viewer and add point cloud and normals-----
	// --------------------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 3, 0.1, "normals");
	//viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}
void CReconstruction3D::rejectPoint3D(vector<Point2d>& vimgPointsInlier1, vector<Point2d>& vimgPointsInlier2, vector<Point3d> &vecReconstruction3D){
	double avgx = 0, avgy = 0, avgz = 0, avgdevx = 0, avgdevy = 0, avgdevz = 0;
	int numelement = vecReconstruction3D.size();
	//calculate the average
	for (int i = 0; i < numelement; i++){
		avgx += vecReconstruction3D[i].x;
		avgy += vecReconstruction3D[i].y;
		avgz += vecReconstruction3D[i].z;
	}

	avgx /= numelement;
	avgy /= numelement;
	avgz /= numelement;
	for (int i = 0; i < numelement; i++){
		avgdevx += abs(vecReconstruction3D[i].x-avgx);
		avgdevy += abs(vecReconstruction3D[i].y-avgy);
		avgdevz += abs(vecReconstruction3D[i].z-avgz);
	}
	avgdevx /= numelement;
	avgdevy /= numelement;
	avgdevz /= numelement;
	/*vector<Point2d> vimgPointsInlier11 = vimgPointsInlier1;
	vector<Point2d> vimgPointsInlier22 = vimgPointsInlier2;
	vector<Point3d> vecReconstruction3D1 = vecReconstruction3D;*/
	//delete corresponding points and 3D point
	float standarddev = 1.75;
	for (int i = 0; i < vecReconstruction3D.size(); i++){
		if ((abs(vecReconstruction3D[i].z - avgz) > standarddev * abs(avgdevz))){// || (abs(vecReconstruction3D[i].x - avgx) > standarddev * abs(avgdevx)) || (abs(vecReconstruction3D[i].y - avgy) > standarddev * abs(avgdevy))){
			//cout << "i " << i << " p1 size " << vimgPointsInlier1.size() << " 3D size " << vecReconstruction3D.size() << endl;
			vecReconstruction3D.erase(vecReconstruction3D.begin() + i);
			vimgPointsInlier1.erase(vimgPointsInlier1.begin() + i);
			vimgPointsInlier2.erase(vimgPointsInlier2.begin() + i);
			i--;
		}
		if (vecReconstruction3D.size() < numelement / 2)
			break;
	}
}
void CReconstruction3D::point3DReconstruct(Mat& P, Mat& P1, vector<Point2d>& vimgPointsInlier1, vector<Point2d>& vimgPointsInlier2, Mat& matReconstruct3D){
	Mat tmpImg;
	triangulatePoints(P, P1, vimgPointsInlier1, vimgPointsInlier2, matReconstruct3D);
	tmpImg = matReconstruct3D.row(3).clone();
	vconcat(tmpImg, tmpImg, tmpImg);
	vconcat(tmpImg, tmpImg, tmpImg);
	divide(matReconstruct3D, tmpImg, matReconstruct3D);

}

bool CReconstruction3D::checkCoherentRotation(Mat& R){
	if (fabsf(determinant(R)) - 1.0 > 1e-7)
		return false;
	return true;
}

double CReconstruction3D::triangulatePoints1(const vector<Point2d>& vImgPoints1, const vector<Point2d>& vImgPoints2, const Mat& intrinsic, Mat& P, Mat& P1, vector<Point3d>& pointClound){
	pointClound.clear();
	vector<double> reprojectionError;
	Point2d kp;
	double me = 0;
	//calculate the real coordinate of each point
	for (int i = 0; i < vImgPoints1.size(); i++){
		//get the first 2D point
		kp = vImgPoints1[i];
		Point3d u(kp.x, kp.y, 1.0);
		Mat um = intrinsic.inv()*Mat(u);
		u = Point3d(um.at<double>(0, 0), um.at<double>(1, 0), um.at<double>(2, 0));

		//get the second 2D point
		kp = vImgPoints2[i];
		Point3d u1(kp.x, kp.y, 1.0);
		Mat um1 = intrinsic.inv()*Mat(u1);
		u1 = Point3d(um1.at<double>(0, 0), um1.at<double>(1, 0), um1.at<double>(2, 0));

		Mat X = linearLSTriangulation(u, P, u1, P1);

		pointClound.push_back(Point3d(X.at<double>(0, 0), X.at<double>(1, 0), X.at<double>(2, 0)));
	}
	return me;
}

Mat CReconstruction3D::linearLSTriangulation(Point3d& u, Mat& P, Point3d& u1, Mat& P1){
	//solve triangulation problem
	Mat A, B, X;
	Mat tmp, tmpA;
	tmpA = Mat(4, 4, CV_64FC1);
	tmp = u.x*P.row(2) - P.row(0);
	tmp.copyTo(tmpA.row(0));
	tmp = u.y*P.row(2) - P.row(1);
	tmp.copyTo(tmpA.row(1));

	tmp = u1.x*P1.row(2) - P1.row(0);
	tmp.copyTo(tmpA.row(2));
	tmp = u1.y*P1.row(2) - P1.row(1);
	tmp.copyTo(tmpA.row(3));

	A = tmpA.colRange(0, 3).clone();
	B = tmpA.col(3).clone()*(-1);

	//solve for X
	solve(A, B, X, DECOMP_SVD);

	return X;
}

float CReconstruction3D::ComputeDFScore(const Mat& mInput8UC1){

	Mat src32F = Mat(mInput8UC1.rows, mInput8UC1.cols, CV_32FC1);
	Mat dst32F = Mat(mInput8UC1.rows, mInput8UC1.cols, CV_32FC1);
	mInput8UC1.convertTo(src32F, CV_32F);

	HaarWavelet(src32F, dst32F, 1);

	Mat diagonal;
	diagonal = dst32F(Range(dst32F.rows / 2 + 2, dst32F.rows - 2), Range(dst32F.cols / 2 + 2, dst32F.cols - 2)).clone();

	Scalar scMean, scStd;
	meanStdDev(diagonal, scMean, scStd);

	float fScore;
	fScore = exp(-scStd.val[0]);

	return fScore;
}

float CReconstruction3D::ComputeDFScore2(const Mat& mInput8UC1){

	cv::Mat M = (Mat_<double>(3, 1) << -1, 2, -1);
	cv::Mat G = cv::getGaussianKernel(3, -1, CV_64F);

	cv::Mat Lx;
	cv::sepFilter2D(mInput8UC1, Lx, CV_64F, M, G);

	cv::Mat Ly;
	cv::sepFilter2D(mInput8UC1, Ly, CV_64F, G, M);

	cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

	double focusMeasure = cv::mean(FM).val[0];
	return (float)focusMeasure;
}


void CReconstruction3D::HaarWavelet(Mat &src, Mat &dst, int NIter){
	/**
	* Parameters:
	\n	+src: the single channel image to perform the Haar wavelet transform.
	\n	+NIter: number of layer in the pyarimid
	\n\n Perform the Haar-wavelet transform.
	*/
	float dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;
	for (int k = 0; k < NIter; k++)
	{
		for (int y = 0; y < (height >> (k + 1)); y++)
		{
			for (int x = 0; x < (width >> (k + 1)); x++)
			{

				dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
			}
		}
		dst.copyTo(src);
	}
}

Mat CReconstruction3D::insertVirtualObjects(){
	m_nCounter++;
	// Project the 3D objects to 2D
	VirtualObject& vobj = algParams.vVirObjects[0];

	if (m_bUseKalmanTracker){
		// Correct the Kalman filters
		Mat R, T, predictR(algParams.R.size(), CV_64F), predictT(algParams.R.size(), CV_64F);
		R = m_camPoseKF_R.predict();
		T = m_camPoseKF_T.predict();

		//cout << "# of inliers: " << m_nSizeOfInliners << endl;
		if (m_nSizeOfInliners < m_nMinSizeOfInliersThresh){
			// Use the prediction value instead of the computed R, T
			R(Range(0, 3), Range(0, 1)).copyTo(predictR);
			T(Range(0, 3), Range(0, 1)).copyTo(predictT);
			R = m_camPoseKF_R.correct(predictR);
			T = m_camPoseKF_T.correct(predictT);
		}
		else{
			R = m_camPoseKF_R.correct(algParams.R);
			T = m_camPoseKF_T.correct(algParams.T);
		}
		R(Range(0, 3), Range(0, 1)).copyTo(algParams.R_KF);
		T(Range(0, 3), Range(0, 1)).copyTo(algParams.T_KF);

		//cout << "R\n" << algParams.R << endl;
		//cout << "KF R\n" << algParams.R_KF << endl;
		//cout << "T\n" << algParams.T << endl;
		//cout << "KF T\n" << algParams.T_KF << endl;

		if (m_nCounter > 5)
		{
			projectPoints(vobj.vertices3D, algParams.R_KF, algParams.T_KF, algParams.intrinsic,
				algParams.distcoeff, vobj.vertices2D);//here is the problem, probably.
		}
		else
			projectPoints(vobj.vertices3D, algParams.R, algParams.T, algParams.intrinsic,
			algParams.distcoeff, vobj.vertices2D);//here is the problem, probably.
	}
	else
	{
		projectPoints(vobj.vertices3D, algParams.R, algParams.T, algParams.intrinsic,
			algParams.distcoeff, vobj.vertices2D);//here is the problem, probably.
	}

	//cout << "algParams.R\n" << algParams.R << endl;
	//cout << "algParams.T\n" << algParams.T << endl;
	algParams.img2.copyTo(m_mDisplay);
	drawVirObjectOnImage(m_mDisplay, vobj);
	imshow("Virtual objects", m_mDisplay);

	return m_mDisplay;
};

void CReconstruction3D::viewGeometry2D(){
	algParams.img2.copyTo(m_mDisplay);
	drawVirObjectOnImage(m_mDisplay, m_delaunyGeo, 3);
	//imshow("Geometry", m_mDisplay);
	cvWaitKey();
}

void CReconstruction3D::estimateSceneBBox(){

	m_sBoundingBox3D.dMinX = m_sBoundingBox3D.dMaxX = algParams.vecPointCloud[0].vecPoint3D[0].x;
	m_sBoundingBox3D.dMinY = m_sBoundingBox3D.dMaxY = algParams.vecPointCloud[0].vecPoint3D[0].y;
	m_sBoundingBox3D.dMinZ = m_sBoundingBox3D.dMaxZ = algParams.vecPointCloud[0].vecPoint3D[0].z;
	for (int j = 0; j < algParams.vecPointCloud[0].vecPoint3D.size(); j++){
		m_sBoundingBox3D.dMinX = min(m_sBoundingBox3D.dMinX, algParams.vecPointCloud[0].vecPoint3D[j].x);
		m_sBoundingBox3D.dMinY = min(m_sBoundingBox3D.dMinY, algParams.vecPointCloud[0].vecPoint3D[j].y);
		m_sBoundingBox3D.dMinZ = min(m_sBoundingBox3D.dMinZ, algParams.vecPointCloud[0].vecPoint3D[j].z);
		m_sBoundingBox3D.dMaxX = max(m_sBoundingBox3D.dMaxX, algParams.vecPointCloud[0].vecPoint3D[j].x);
		m_sBoundingBox3D.dMaxY = max(m_sBoundingBox3D.dMaxY, algParams.vecPointCloud[0].vecPoint3D[j].y);
		m_sBoundingBox3D.dMaxZ = max(m_sBoundingBox3D.dMaxZ, algParams.vecPointCloud[0].vecPoint3D[j].z);
	}
	
};

void CReconstruction3D::generateVirtualObject(){
	// estimate the bounding box of scene from the point cloud
	estimateSceneBBox();

	float fBoxSize = (float)(m_sBoundingBox3D.dMaxX - m_sBoundingBox3D.dMinX) / 10.;
	VirtualObject vobj;
	// Add vertices
	Point3f pt;

	m_prev3DPointsMass.x = (float)(m_sBoundingBox3D.dMaxX + m_sBoundingBox3D.dMinX) / 2.f;
	m_prev3DPointsMass.y = (float)(m_sBoundingBox3D.dMaxY + m_sBoundingBox3D.dMinY) / 2.f;
	m_prev3DPointsMass.z = (float)(m_sBoundingBox3D.dMaxZ + m_sBoundingBox3D.dMinZ) / 2.f;

	// Find the nearest point to attach the object to
	int nIdx = 0;
	double dMinDis = DBL_MAX, dVal;
	for (int j = 0; j < algParams.vecPointCloud[0].vecPoint3D.size(); j++){
		dVal = ED(algParams.vecPointCloud[0].vecPoint3D[j], m_prev3DPointsMass);
		if (dVal < dMinDis){
			dMinDis = dVal;
			nIdx = j;
		}
	}

	pt = algParams.vecPointCloud[0].vecPoint3D[nIdx];
	float fSignX = -1, fSignY = 1, fSignZ = 1;
	vobj.vertices3D.push_back(pt);
	vobj.vertices3D.push_back(pt + Point3f(fSignX*fBoxSize, 0, 0));
	vobj.vertices3D.push_back(pt + Point3f(fSignX*fBoxSize, fSignY*fBoxSize, 0));
	vobj.vertices3D.push_back(pt + Point3f(0, fSignY*fBoxSize, 0));

	vobj.vertices3D.push_back(pt + Point3f(0, 0, fSignZ*fBoxSize));
	vobj.vertices3D.push_back(pt + Point3f(0, fSignY*fBoxSize, fSignZ*fBoxSize));
	vobj.vertices3D.push_back(pt + Point3f(fSignX*fBoxSize, fSignY*fBoxSize, fSignZ*fBoxSize));
	vobj.vertices3D.push_back(pt + Point3f(fSignX*fBoxSize, 0, fSignZ*fBoxSize));
	// Add faces
	vobj.faces.resize(6);
	vobj.faces[0].push_back(0); vobj.faces[0].push_back(1); vobj.faces[0].push_back(2); vobj.faces[0].push_back(3);
	vobj.faces[1].push_back(4); vobj.faces[1].push_back(5); vobj.faces[1].push_back(6); vobj.faces[1].push_back(7);
	vobj.faces[2].push_back(0); vobj.faces[2].push_back(4); vobj.faces[2].push_back(5); vobj.faces[2].push_back(3);
	vobj.faces[3].push_back(3); vobj.faces[3].push_back(5); vobj.faces[3].push_back(6); vobj.faces[3].push_back(2);
	vobj.faces[4].push_back(2); vobj.faces[4].push_back(6); vobj.faces[4].push_back(1); vobj.faces[4].push_back(7);
	vobj.faces[5].push_back(1); vobj.faces[5].push_back(7); vobj.faces[5].push_back(4); vobj.faces[5].push_back(0);
	algParams.vVirObjects.push_back(vobj);

	// Visualize the object in point cloud
	vector<Point3d> v3D;
	for (int i = 0; i < vobj.vertices3D.size(); i++)
		v3D.push_back(vobj.vertices3D[i]);
	algParams.vecRegisteredPoints3D.clear();
	algParams.vecRegisteredPoints3D.push_back(algParams.vecPointCloud[0].vecPoint3D);
	algParams.vecRegisteredPoints3D.push_back(v3D);
};

float CReconstruction3D::reprojectionError(vector<int> & inliers, Mat & rotationVector, Mat & translationVector, vector<Point3d> & points3D_p, vector<Point2d> & points2D){
	vector<Point3f> points3D;
	//change things to Point3f, may not be necessary
	for (int i = 0; i < points3D_p.size(); i++){
		points3D.push_back(Point3f(points3D_p[i].x, points3D_p[i].y, points3D_p[i].z));
	}
	vector<Point2f> imagepoints; 
	//project the points
	projectPoints(points3D, rotationVector, translationVector, algParams.intrinsic, algParams.distcoeff, imagepoints);
	float err = 0;
	//calculate the reprojection error for each point
	for (int i = 0; i < points2D.size(); i++){
		float er = sqrt(pow(points2D[i].x - imagepoints[i].x, 2) + pow(points2D[i].y - imagepoints[i].y, 2));
		if (er <= 3.0){
			err += er;
			inliers.push_back(i);
		}
	}
	if (inliers.size() == 0)
		return RAND_MAX;
	return err / inliers.size();
}

vector<int> CReconstruction3D::PnPRansac(vector<Point3d>  & prevPoints3D, vector<Point2d> & nextPoints){
	vector<int> bestinliers, nextbestinliers;
	Mat BestR, BestT, NextBestR, NextBestT;
	int nextbest = 0;
	float besterr = 10000000000;
	int minimuminliers = nextPoints.size() / 2;
	int iterations = 8;
	for (int i = 0; i < iterations; i++){
		vector<Point3f> inliers3D;
		vector<Point2f> inlierPoints;
		vector<int> inliers;
		Mat translationVector = algParams.T.clone();
		Mat rotationVector = algParams.R.clone();
		//find six points, this may be slightly less than completely random
		vector<int> previndex;
		for (int j = 0; j < 6; j++){
			int index = rand() % nextPoints.size();
			bool done = false;
			while (!done){
				done = true;
				for (int k = 0; k < previndex.size(); k++){
					if (index == previndex[k])
						index = (index + 1) % nextPoints.size();
				}
			}
			previndex.push_back(index);
			inliers3D.push_back(Point3f(prevPoints3D[index].x, prevPoints3D[index].y, prevPoints3D[index].z));
			inlierPoints.push_back(Point2f(nextPoints[index].x, nextPoints[index].y));
		}
		// see if we can use extrinsic guess in the future
		solvePnP(inliers3D, inlierPoints, algParams.intrinsic, algParams.distcoeff, rotationVector, translationVector, false, CV_ITERATIVE);
		float err = reprojectionError(inliers, rotationVector, translationVector, prevPoints3D, nextPoints); //average error 
		//if we have enough inliers, save it to bestpoints
		if (inliers.size() > minimuminliers){
			if (besterr > err){
				rotationVector.copyTo(BestR);
				translationVector.copyTo(BestT);
				besterr = err;
				bestinliers = inliers;
			}
		}
		else if (!bestinliers.size()){//see if it's the best option so far
			if (inliers.size() > nextbest){
				nextbest = inliers.size();
				rotationVector.copyTo(NextBestR);
				translationVector.copyTo(NextBestT);
				nextbestinliers = inliers;
			}
		}

	}
	
	if (bestinliers.size()){//if we found enough points to base camera position off of
		vector<Point3f> inliers3D;
		vector<Point2f> inlierPoints;
		vector<int> inliers;
		//Use all the inliers to calculate position
		vector<int> previndex;
		for (int j = 0; j < bestinliers.size(); j++){
			inliers3D.push_back(Point3f(prevPoints3D[bestinliers[j]].x, prevPoints3D[bestinliers[j]].y, prevPoints3D[bestinliers[j]].z));
			inlierPoints.push_back(Point2f(nextPoints[bestinliers[j]].x, nextPoints[bestinliers[j]].y));
		}
		algParams.R.copyTo(m_mPrevR);
		algParams.T.copyTo(m_mPrevT);
		solvePnP(inliers3D, inlierPoints, algParams.intrinsic, algParams.distcoeff, algParams.R, algParams.T, false, CV_ITERATIVE);
	}
	else{//momentum function, try to estimate where the camera will be from previous frames
		Mat translationVector = algParams.T.clone();
		Mat rotationVector = algParams.R.clone();
		vector<int> inliers = nextbestinliers;
		for (int j = 0; j < 1; j++){
			vector<Point3d> inliers3D;
			vector<Point2d> inlierPoints;
			
			for (int i = 0; i < inliers.size(); i++){
				inliers3D.push_back(prevPoints3D[inliers[i]]);
				inlierPoints.push_back(nextPoints[inliers[i]]);
			}
			inliers.clear();
			if (inliers3D.size() >= 6){
				solvePnP(inliers3D, inlierPoints, algParams.intrinsic, algParams.distcoeff, rotationVector, translationVector, false, CV_ITERATIVE);//Four points
				float err = reprojectionError(inliers, rotationVector, translationVector, prevPoints3D, nextPoints); //average error
				//cout << "inliers.size() " << inliers.size() << endl;
			}
		}
		if (inliers.size() > nextPoints.size() / 2){
			algParams.R.copyTo(m_mPrevR);
			algParams.T.copyTo(m_mPrevT);
			//cout << "rotationVector\n" << rotationVector << endl;
			//cout << "translationVector\n" << translationVector << endl;
			//cout << "algParams.T\n" << algParams.T << endl;
			//cout << "translationVector\n" << translationVector << endl;
			//cout << "algParams.R\n" << algParams.R << endl;
			cout << "rotationVector\n" << rotationVector << endl;
			//cout << "bestinliers.size() (second round) " << bestinliers.size() << " total " << nextPoints.size() << endl;
			rotationVector.copyTo(algParams.R);
			translationVector.copyTo(algParams.T);
			
		}
		else{
			Mat tempmat;
			algParams.R.copyTo(tempmat);
			algParams.R += algParams.R - m_mPrevR;
			tempmat.copyTo(m_mPrevR);

			algParams.T.copyTo(tempmat);
			algParams.T += algParams.T - m_mPrevT;
			tempmat.copyTo(m_mPrevT);
		}
	}
	return bestinliers;
}

void CReconstruction3D::estimateCamPose(){
	// Use the successfully tracked points to estimate the camera pose
	vector<Point2d> imagePts;
	vector<Point3d> worldPts;
	//remove bad lines using epipolar something
	
	for (int i = 0; i < m_vecOFTrackStatus.size(); i++){
		if (m_vecOFTrackStatus[i] && m_vecImgPoints2F[i].x >= 0 && m_vecImgPoints2F[i].y >= 0 &&//and make sure the points are within the frame
			m_vecImgPoints2F[i].x <= m_nImgWidth && m_vecImgPoints2F[i].y <= m_nImgHeight){
			worldPts.push_back(m_sLastPointCloud.vecPoint3D[i]);
			imagePts.push_back(m_vecImgPoints2F[i]);
		}
	}

	m_nSizeOfInliners = (int)imagePts.size();
	//if (m_nSizeOfInliners < m_nMinSizeOfInliersThresh){
	//	algParams.bIsKeyFrame = true;
	//	return;
	//}

	if ((int)imagePts.size() < 6){
		Mat tempmat;
		algParams.R.copyTo(tempmat);
		algParams.R += algParams.R - m_mPrevR;
		tempmat.copyTo(m_mPrevR);

		algParams.T.copyTo(tempmat);
		algParams.T += algParams.T - m_mPrevT;
		tempmat.copyTo(m_mPrevT);
		return;
	}

	//All of the above is fine
	//solvePnP(worldPts, imagePts, algParams.intrinsic, algParams.distcoeff, algParams.R, algParams.T, false, 0);
	PnPRansac(worldPts, imagePts);//this positions the camera
}

void CReconstruction3D::drawVirObjectOnImage(Mat& image, VirtualObject& vObj, int nPolySize){
	Scalar color(0, 0, 255);
	vector<vector<int>>& vfaces = vObj.faces;
	vector<Point2f>& vpoints2D = vObj.vertices2D;
	if (nPolySize == 4)
		for (int i = 0; i < (int)vfaces.size(); i++){
			line(image, vpoints2D[vfaces[i][0]], vpoints2D[vfaces[i][1]], color);
			line(image, vpoints2D[vfaces[i][1]], vpoints2D[vfaces[i][2]], color);
			line(image, vpoints2D[vfaces[i][2]], vpoints2D[vfaces[i][3]], color);
			line(image, vpoints2D[vfaces[i][3]], vpoints2D[vfaces[i][0]], color);
		}

	if (nPolySize == 3)
		for (int i = 0; i < (int)vfaces.size(); i++){
			line(image, vpoints2D[vfaces[i][0]], vpoints2D[vfaces[i][1]], color);
			line(image, vpoints2D[vfaces[i][1]], vpoints2D[vfaces[i][2]], color);
			line(image, vpoints2D[vfaces[i][2]], vpoints2D[vfaces[i][0]], color);
		}
}

void CReconstruction3D::trackToEstimatePose(){//tracking to place the virtual object
	/// Parameters for Shi-Tomasi algorithm
	//cvtColor(algParams.img1, m_mGrayImg1, CV_RGB2GRAY);
	cvtColor(algParams.img2, m_mGrayImg2, CV_RGB2GRAY);

	//vector<Point2f> c1, c2;
	vector<float> err;
	Size winSize(m_sFundamentalParams.nWinSize, m_sFundamentalParams.nWinSize);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.001);

	m_vecImgPoints2F.clear();
	calcOpticalFlowPyrLK(m_mGrayImg1, m_mGrayImg2, m_vecLastSeedFrame2D, m_vecImgPoints2F,
		m_vecOFTrackStatus, err, winSize, 4, termcrit, 0);//find points again in new image

	int nCnt = 0;
	for (int i = 0; i < m_vecOFTrackStatus.size(); i++){//this stops tracking a point if the optical flow error is too large
		//cout << "trackStatus[" << i << "] " << (int)m_vecTrackStatus[i] << " err[" << i << "] " << err[i] << " m_vecImgPoints2F[" << i << "] " << m_vecImgPoints2F[i] << " m_vecLastSeedFrame2D[" << i << "] " << m_vecLastSeedFrame2D[i] << endl;
		if (m_vecOFTrackStatus[i]){
			if (err[i] > m_sFundamentalParams.fErr) {
				m_vecOFTrackStatus[i] = 0;
				continue;
			}
			Point2d pt = m_vecImgPoints2F[i];
			if ((pt.x < 0) || (pt.y < 0) || (pt.x >= m_nImgWidth) || (pt.y >= m_nImgHeight)){
				m_vecOFTrackStatus[i] = 0;
				continue;
			}
			nCnt++;
		}
	}
	m_vecInliners1F.resize(nCnt);
	m_vecInliners2F.resize(nCnt);

	for (int i = 0, j= 0; i < m_vecOFTrackStatus.size();i++)
		if (m_vecOFTrackStatus[i]){
			m_vecInliners1F[j] = m_vecLastSeedFrame2D[i];
			m_vecInliners2F[j++] = m_vecImgPoints2F[i];
		}


	//rejectOutlier(m_vecInliners1F, m_vecInliners2F, m_vecInlinerStatus, 0.005, 0.99);
	rejectOutlier(m_vecInliners1F, m_vecInliners2F, m_vecInlinerStatus, 0.001, 0.99);

	// For display purpose only
	for (int i = m_vecInlinerStatus.size() - 1; i >= 0;i--)
		if (!m_vecInlinerStatus[i]){
			m_vecInliners1F.erase(m_vecInliners1F.begin() + i);
			m_vecInliners2F.erase(m_vecInliners2F.begin() + i);
		}
	drawMatchingPoints(m_mGrayImg1, algParams.img2, m_mCanvas, m_vecInliners1F, m_vecInliners2F);//draws points in second window

	//this makes more sense here
	estimateCamPose();

	// check if it is a key frame for a new point cloud
	selectKeyFrame(m_vecLastSeedFrame2D, m_vecImgPoints2F, m_vecOFTrackStatus);
}
bool CReconstruction3D::rotationCheck(vector<Point2f> m_vecInliners1F, vector<Point2f> m_vecInliners2F){
	vector<float> fAngle;
	for (int i = 0; i < m_vecInliners1F.size(); i++)
		fAngle.push_back(ComputeAngle(m_vecInliners1F[i], m_vecInliners2F[i]));
	float fMeanAngle = 0;
	for (int i = 0; i < fAngle.size(); i++)
		fMeanAngle += fAngle[i];
	fMeanAngle /= (float)fAngle.size();
	cout << "fMeanAngle: " << fMeanAngle << endl;

	if ((fMeanAngle > 80) && (fMeanAngle < 100))
		return false;
	else
		return true;
}

	float CReconstruction3D::ComputeAngle(Point2f p1, Point2f p2)
	{
		double de = ED(p1, p2);
		float ang_deg, ang_rad;
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
		return ang_deg;
	}


void CReconstruction3D::rebase(){
	m_nFramesFromBase++;
	//maybe if the Ransac does poorly, it throws a flag?

	//Mat m = m_mSeedFrameT - algParams.T;
	//pow(m, 2.0, m);
	//reduce(m, m, 0, CV_REDUCE_SUM);
	//sqrt(m, m);
	//Mat r = m_mSeedFrameR - algParams.R;
	//double min, max;
	//minMaxLoc(abs(r), &min, &max);

	//pow(r, 2.0, r);
	//reduce(r, r, 0, CV_REDUCE_SUM);
	//sqrt(r, r);

	//cout << "Rotated amount:		" << r.at<double>(0) << "		Max:	" << max << endl;
	//if ((m.at<double>(0) > .7 && r.at<double>(0) > .3) || m.at<double>(0) > 1.5)
	{//see if we've moved far enough, or rotated far enough

		//create camera matrices
		Mat rotationArray;
		Rodrigues(algParams.R, rotationArray);
		rotationArray = rotationArray.t();
		rotationArray.resize(4);
		rotationArray = rotationArray.t();
		algParams.T.copyTo(rotationArray.col(3));
		Mat P, P1;

		
		m_vecPositionMatricies.push_back(rotationArray);
	
		/// develop a module to check if the movement is moving forward or turning
		// if the movement is going forward then do reconstruction, otherwise no
		rotationCheck(m_vecInliners1F, m_vecInliners2F); // return YES is moving forward; NO if turning
			constructPointCloudInReverse();

		if (algParams.nStatus == ERR_REBASE){
			m_vecPositionMatricies.erase(m_vecPositionMatricies.begin() + m_vecPositionMatricies.size()-1);
			return;
		}

		addDescriptorsToPC();

		//Reset how many frames we're off-base
		//set the values for the new seed frame
		algParams.R.copyTo(m_mSeedFrameR);
		algParams.T.copyTo(m_mSeedFrameT);
		algParams.img2.copyTo(algParams.img1);
		cvtColor(algParams.img1, m_mGrayImg1, CV_BGR2GRAY);
		//algParams.vecPoints3D.push_back(m_sLastPointCloud.vecPoint3D);
		algParams.R.copyTo(m_sLastPointCloud.mR);
		algParams.T.copyTo(m_sLastPointCloud.mT);
		algParams.vecPointCloud.push_back(m_sLastPointCloud);
		m_nFramesFromBase = 0;

		//Keyframe* pKeyframe = new Keyframe( &algParams.vecPointCloud.back() );
		//m_keyFrames.add( pKeyframe );
	}
}

void CReconstruction3D::init4Tracker(){
	// Initialize Rotation Kalman filter
	m_camPoseKF_R = KalmanFilter(6, 3, 0, CV_64F);
	for (int i = 0; i < 6;i++)
		m_camPoseKF_R.transitionMatrix.at<double>(i, i) = 1;
	m_camPoseKF_R.transitionMatrix.at<double>(0, 3) = 1;
	m_camPoseKF_R.transitionMatrix.at<double>(1, 4) = 1;
	m_camPoseKF_R.transitionMatrix.at<double>(2, 5) = 1;
	m_camPoseKF_R.statePre.setTo(0);
	m_camPoseKF_R.statePre.at<double>(0) = algParams.R.at<double>(0);
	m_camPoseKF_R.statePre.at<double>(1) = algParams.R.at<double>(1);
	m_camPoseKF_R.statePre.at<double>(2) = algParams.R.at<double>(2);
	setIdentity(m_camPoseKF_R.measurementMatrix);
	setIdentity(m_camPoseKF_R.processNoiseCov, Scalar::all(1e-5));
	setIdentity(m_camPoseKF_R.measurementNoiseCov, Scalar::all(1e-3));
	setIdentity(m_camPoseKF_R.errorCovPost, Scalar::all(.01));

	// Initialize Translation Kalman filter
	m_camPoseKF_T = KalmanFilter(6, 3, 0, CV_64F);
	for (int i = 0; i < 6; i++)
		m_camPoseKF_T.transitionMatrix.at<double>(i, i) = 1;
	m_camPoseKF_T.transitionMatrix.at<double>(0, 3) = 1;
	m_camPoseKF_T.transitionMatrix.at<double>(1, 4) = 1;
	m_camPoseKF_T.transitionMatrix.at<double>(2, 5) = 1;
	m_camPoseKF_T.statePre.at<double>(0) = algParams.T.at<double>(0);
	m_camPoseKF_T.statePre.at<double>(1) = algParams.T.at<double>(1);
	m_camPoseKF_T.statePre.at<double>(2) = algParams.T.at<double>(2);
	setIdentity(m_camPoseKF_T.measurementMatrix);
	setIdentity(m_camPoseKF_T.processNoiseCov, Scalar::all(1e-5));
	setIdentity(m_camPoseKF_T.measurementNoiseCov, Scalar::all(1e-4));
	setIdentity(m_camPoseKF_T.errorCovPost, Scalar::all(.01));

	algParams.R_KF.create(algParams.R.size(), CV_64F);
	algParams.T_KF.create(algParams.T.size(), CV_64F);
}

void CReconstruction3D::constructFirstPointCloud(){
	m_vecPositionMatricies.clear();

	Mat matReconstruct3D;
	//vector<Point3d> vecReconstruction3D;

	// Estimate the fundamental matrix
	estimateFundamentalMatrix();

	if (algParams.nStatus == ERR_FIND_CAM_MATRIX){
		return;
	}

	m_vecPositionMatricies.push_back(algParams.P);
	m_vecPositionMatricies.push_back(algParams.P1);

	constructPointCloudInReverse();

	addDescriptorsToPC();

	//set the values for the new seed frame
	algParams.R.copyTo(m_mSeedFrameR);
	algParams.T.copyTo(m_mSeedFrameT);
	algParams.img2.copyTo(algParams.img1);
	cvtColor(algParams.img1, m_mGrayImg1, CV_BGR2GRAY);
	//algParams.vecPoints3D.push_back(m_sLastPointCloud.vecPoint3D);
	algParams.R.copyTo(m_sLastPointCloud.mR);
	algParams.T.copyTo(m_sLastPointCloud.mT);
	algParams.vecPointCloud.push_back(m_sLastPointCloud);
}

void CReconstruction3D::addDescriptorsToPC(){
	// Screening the inlier points on image2 to define the descriptors
	for (int i = m_vecOFTrackStatus.size()-1; i >= 0; i--)
		if (m_vecOFTrackStatus[i] == 0)
			m_vecKeypoints2.erase(m_vecKeypoints2.begin() + i);

	//m_pDescExtractor->compute(m_mGrayImg1, m_vecKeypoints1, m_mDescriptor1);
	m_pDescExtractor->compute(m_mGrayImg2, m_vecKeypoints2, m_sLastPointCloud.matDescriptors);
	//
	//vector< vector<DMatch> > matches;

	//m_vecMatched1.clear();
	//m_vecMatched2.clear();
	//
	//double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
	//double ransac_thresh = 3.5f; // RANSAC inlier threshold
	//double nn_match_ratio = 0.8f; // Nearest-neighbor matching ratio


	//m_pDescMatcher->knnMatch(m_mDescriptor1, m_mDescriptor2, matches, 2);
	//for (unsigned i = 0; i < matches.size(); i++) {
	//	if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
	//		m_vecMatched1.push_back(m_vecKeypoints1[matches[i][0].queryIdx]);
	//		m_vecMatched2.push_back(m_vecKeypoints2[matches[i][0].trainIdx]);
	//	}
	//}


	//Mat inlier_mask, homography;
	//vector<KeyPoint> inliers1, inliers2;
	//vector<DMatch> inlier_matches;
	//Mat m = Mat(m_vecMatched1);
	//
	//KeyPoint::convert(m_vecMatched1, m_vecImgPoints1F);
	//KeyPoint::convert(m_vecMatched2, m_vecImgPoints2F);

	//if (m_vecMatched1.size() >= 4) {
	//	homography = findHomography(m_vecImgPoints1F, m_vecImgPoints2F,
	//		RANSAC, ransac_thresh, inlier_mask);
	//}

	//if (m_vecMatched1.size() < 4 || homography.empty()) {
	//	Mat res;
	//	hconcat(algParams.img1, algParams.img2, res);
	//}

	//for (unsigned i = 0; i < m_vecMatched1.size(); i++) {
	//	if (inlier_mask.at<uchar>(i))
	//	{
	//		int new_i = static_cast<int>(inliers1.size());
	//		inliers1.push_back(m_vecMatched1[i]);
	//		inliers2.push_back(m_vecMatched2[i]);
	//		inlier_matches.push_back(DMatch(new_i, new_i, 0));
	//	}
	//}
	//
	//drawKeypoints(algParams.img2, m_vecKeypoints1, algParams.img2);

	//drawMatches(algParams.img1, inliers1, algParams.img2, inliers2,	inlier_matches, m_mCanvas, Scalar(255, 0, 0), Scalar(255, 0, 0));
	//imshow("ORB matches", m_mCanvas);
	//cvWaitKey();

	////////////////////////////////////////////////////////////////////////////
	//// Optical flow matches
	//KeyPoint::convert(m_vecKeypoints1, m_vecImgPoints1F);
	//vector<float> err;
	//Size winSize(m_sFundamentalParams.nWinSize, m_sFundamentalParams.nWinSize);
	//TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.001);

	//m_vecImgPoints2F.clear();
	//calcOpticalFlowPyrLK(m_mGrayImg1, m_mGrayImg2, m_vecImgPoints1F, m_vecImgPoints2F,
	//	m_vecOFTrackStatus, err, winSize, 4, termcrit, 0);//find points again in new image

	//int nCnt = 0;
	//for (int i = 0; i < m_vecOFTrackStatus.size(); i++){//this stops tracking a point if the optical flow error is too large
	//	if (m_vecOFTrackStatus[i]){
	//		if (err[i] > m_sFundamentalParams.fErr) {
	//			m_vecOFTrackStatus[i] = 0;
	//			continue;
	//		}
	//		Point2d pt = m_vecImgPoints2F[i];
	//		if ((pt.x < 0) || (pt.y < 0) || (pt.x >= m_nImgWidth) || (pt.y >= m_nImgHeight)){
	//			m_vecOFTrackStatus[i] = 0;
	//			continue;
	//		}
	//		nCnt++;
	//	}
	//}
	//m_vecInliners1F.resize(nCnt);
	//m_vecInliners2F.resize(nCnt);

	//for (int i = 0, j = 0; i < m_vecOFTrackStatus.size(); i++)
	//	if (m_vecOFTrackStatus[i]){
	//		m_vecInliners1F[j] = m_vecImgPoints1F[i];
	//		m_vecInliners2F[j++] = m_vecImgPoints2F[i];
	//	}
	//vector<Point2f> inliner1, inliner2;
	//rejectOutlier(m_vecInliners1F, m_vecInliners2F, inliner1, inliner2, .001, 0.99);

	//drawMatchingPoints(m_mCanvas, inliner1, inliner2);//draws points in second window
	//cvWaitKey();
}

void CReconstruction3D::geoEstimation(vector<Point3d>& pointCloud){

	pcl::PointCloud<pcl::PointXYZ>::Ptr basicPointCloud;

	createPointCloud(pointCloud, basicPointCloud);

	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(basicPointCloud);
	n.setInputCloud(basicPointCloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);
	//* normals should not contain the point normals + surface curvatures

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*basicPointCloud, *normals, *cloud_with_normals);
	//* cloud_with_normals = cloud + normals

	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	//pcl::PolygonMesh triangles;

	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius(1000);

	// Set typical values for the parameters
	gp3.setMu(10);
	gp3.setMaximumNearestNeighbors(300);
	gp3.setMaximumSurfaceAngle(M_PI); // 45 degrees
	gp3.setMinimumAngle(0); // 10 degrees
	gp3.setMaximumAngle(M_PI); // 120 degrees
	gp3.setNormalConsistency(true);

	// Get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(m_polyMesh);

	// Additional vertex information
	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();


	pcl::visualization::PCLVisualizer viewer("triangulation");

	string st;
	st = format("Point Cloud");
	viewer.addPolygonMesh(m_polyMesh, st);

	//black color
	viewer.setBackgroundColor(0, 0, 0, 0);
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, st);

	while (!viewer.wasStopped())
	{
		//display the visualizer until 'q' is pressed
		//viewer.spinOnce();
	}

	// Create the Delauny struct
	int	nVertices = pointCloud.size();
	m_delaunyGeo.vertices3D.resize(nVertices);
	m_delaunyGeo.vertices2D.resize(nVertices);
	for (int i = 0; i < nVertices; i++)
	{
		m_delaunyGeo.vertices2D[i] = m_vImgPointsInlier2[i];
		m_delaunyGeo.vertices3D[i] = pointCloud[i];
	}

	nVertices = m_polyMesh.polygons.size();
	m_delaunyGeo.faces.resize(nVertices);
	for (int j = 0; j < nVertices; j++){
		m_delaunyGeo.faces[j].push_back(m_polyMesh.polygons[j].vertices[0]);
		m_delaunyGeo.faces[j].push_back(m_polyMesh.polygons[j].vertices[1]);
		m_delaunyGeo.faces[j].push_back(m_polyMesh.polygons[j].vertices[2]);
	}

}

//void CReconstruction3D::geoEstimation(vector<vector<Point3d>>& pointCloud){
//
//	pcl::PointCloud<pcl::PointXYZ>::Ptr basicPointCloud;
//
//
//	createPointCloud(pointCloud, basicPointCloud);
//
//	// Normal estimation*
//	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
//	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
//	tree->setInputCloud(basicPointCloud);
//	n.setInputCloud(basicPointCloud);
//	n.setSearchMethod(tree);
//	n.setKSearch(20);
//	n.compute(*normals);
//	//* normals should not contain the point normals + surface curvatures
//
//	// Concatenate the XYZ and normal fields*
//	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
//	pcl::concatenateFields(*basicPointCloud, *normals, *cloud_with_normals);
//	//* cloud_with_normals = cloud + normals
//
//	// Create search tree*
//	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
//	tree2->setInputCloud(cloud_with_normals);
//
//	// Initialize objects
//	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
//	//pcl::PolygonMesh triangles;
//
//	// Set the maximum distance between connected points (maximum edge length)
//	gp3.setSearchRadius(1000);
//
//	// Set typical values for the parameters
//	gp3.setMu(10);
//	gp3.setMaximumNearestNeighbors(300);
//	gp3.setMaximumSurfaceAngle(M_PI); // 45 degrees
//	gp3.setMinimumAngle(0); // 10 degrees
//	gp3.setMaximumAngle(M_PI); // 120 degrees
//	gp3.setNormalConsistency(true);
//
//	// Get result
//	gp3.setInputCloud(cloud_with_normals);
//	gp3.setSearchMethod(tree2);
//	gp3.reconstruct(m_polyMesh);
//
//	// Additional vertex information
//	std::vector<int> parts = gp3.getPartIDs();
//	std::vector<int> states = gp3.getPointStates();
//
//
//	pcl::visualization::PCLVisualizer viewer("triangulation");
//
//	string st;
//	st = format("Point Cloud");
//	viewer.addPolygonMesh(m_polyMesh, st);
//
//	//black color
//	viewer.setBackgroundColor(0, 0, 0, 0);
//	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, st);
//
//	while (!viewer.wasStopped())
//	{
//		//display the visualizer until 'q' is pressed
//		viewer.spinOnce();
//	}
//
//	// Create the Delauny struct
//	int	nVertices = pointCloud.size();
//	m_delaunyGeo.vertices3D.resize(nVertices);
//	m_delaunyGeo.vertices2D.resize(nVertices);
//	for (int i = 0; i < nVertices; i++)
//	{
//		m_delaunyGeo.vertices2D[i] = m_vImgPointsInlier2[i];
//		m_delaunyGeo.vertices3D[i] = pointCloud[i];
//	}
//
//	nVertices = m_polyMesh.polygons.size();
//	m_delaunyGeo.faces.resize(nVertices);
//	for (int j = 0; j < nVertices; j++){
//		m_delaunyGeo.faces[j].push_back(m_polyMesh.polygons[j].vertices[0]);
//		m_delaunyGeo.faces[j].push_back(m_polyMesh.polygons[j].vertices[1]);
//		m_delaunyGeo.faces[j].push_back(m_polyMesh.polygons[j].vertices[2]);
//	}
//
//}

void CReconstruction3D::modifyGeo(){
	// Here you code to implement your idea here
	
}

void CReconstruction3D::SaveDescriptedPCVector(vector<DescriptedPC>& vecPC, string sFilename){
	int nsize = vecPC.size();
	cv::FileStorage storage(sFilename, cv::FileStorage::WRITE);
	storage << "CloudSize" << nsize;
	string sNum;
	Mat mPC;
	string sVarName;
	for (int i = 0; i < nsize; i++){
		cvtVec2Mat(vecPC[i].vecPoint3D, mPC);
		sNum = to_string(i);
		sVarName = "Point3D" + sNum;
		storage << sVarName << mPC;
		sVarName = "Descriptors" + sNum;
		storage << sVarName << vecPC[i].matDescriptors;
		sVarName = "mR" + sNum;
		storage << sVarName << vecPC[i].mR;
		sVarName = "mT" + sNum;
		storage << sVarName << vecPC[i].mT;
	}

	storage.release();
	ofstream ofs("data.dat", ios::out | ios::binary);
	if (!ofs) {
		cout << "Cannot open file.";
		return;
	}
	// Save the size of the point cloud
//	int nsize = vecPC.size();
	ofs.write((char*)(&nsize), sizeof nsize);

	for (int i = 0; i < nsize; i++){
		// Save the 3D points 

		Mat m3D = Mat(vecPC[i].vecPoint3D);
		int type = m3D.type();
		ofs.write((const char*)(&m3D.rows), sizeof(int));
		ofs.write((const char*)(&m3D.cols), sizeof(int));
		ofs.write((const char*)(&type), sizeof(int));
		ofs.write((const char*)(m3D.data), m3D.elemSize() * m3D.total());


		// Save the descriptors
		Mat mDes = vecPC[i].matDescriptors;
		type = mDes.type();
		ofs.write((const char*)(&mDes.rows), sizeof(int));
		ofs.write((const char*)(&mDes.cols), sizeof(int));
		ofs.write((const char*)(&type), sizeof(int));
		ofs.write((const char*)(mDes.data), mDes.elemSize() * mDes.total());


		// Saving the R and T matrices
		Mat mR = vecPC[i].mR;
		type = mR.type();
		ofs.write((const char*)(&type), sizeof(int));
		ofs.write((const char*)(mR.data), mR.elemSize() * mR.total());
		Mat mT = vecPC[i].mT;
		type = mT.type();
		ofs.write((const char*)(&type), sizeof(int));
		ofs.write((const char*)(mT.data), mT.elemSize() * mT.total());
	}

	//out.write((char *)&fnum, sizeof fnum);
	ofs.close();
}

void CReconstruction3D::LoadDescriptedPCVector(vector<DescriptedPC>& vecPC, string sFilename){
	//int nsize;
	//cv::FileStorage storage(sFilename, cv::FileStorage::READ);
	//nsize = (int)storage["CloudSize"];
	//DescriptedPC desPC;
	//Mat mPC;
	//string sNum;
	//string sVarName;
	//vecPC.resize(nsize);
	//for (int i = 0; i < nsize; i++){
	//	sNum = to_string(i);
	//	sVarName = "Point3D" + sNum;
	//	storage[sVarName] >> mPC;
	//	cvtMat2Vec(mPC, desPC.vecPoint3D);
	//	sVarName = "Descriptors" + sNum;
	//	storage[sVarName] >> desPC.matDescriptors;
	//	sVarName = "mR" + sNum;
	//	storage[sVarName] >> desPC.mR;
	//	sVarName = "mT" + sNum;
	//	storage[sVarName] >> desPC.mT;
	//	vecPC[i] = desPC;
	//}
	//storage.release();

	int nPCSize;
	ifstream ifs("data.dat", ios::in | ios::binary);

	// Read the size of point cloud
	ifs.read((char *)&nPCSize, sizeof nPCSize);
	vecPC.resize(nPCSize);
	vector<double> v2;
	int nElements;
	int rows, cols, type;
	Mat mTemp;
	for (int i = 0; i < nPCSize; i++){
		//Read the 3D points mat
		ifs.read((char*)(&rows), sizeof(int));
		ifs.read((char*)(&cols), sizeof(int));
		ifs.read((char*)(&type), sizeof(int));
		mTemp.release();
		mTemp.create(rows, cols, type);
		ifs.read((char*)(mTemp.data), mTemp.elemSize() * mTemp.total());
		vector<Point3d> v3d(mTemp);
		vecPC[i].vecPoint3D = v3d;

		//Read the descriptors mat
		Mat& mDes = vecPC[i].matDescriptors;
		ifs.read((char*)(&rows), sizeof(int));
		ifs.read((char*)(&cols), sizeof(int));
		ifs.read((char*)(&type), sizeof(int));
		mDes.create(rows, cols, type);
		ifs.read((char*)(mDes.data), mDes.elemSize() * mDes.total());

		// Read the R and T
		rows = 3;
		cols = 1;
		ifs.read((char*)(&type), sizeof(int));
		mTemp.create(rows, cols, type);
		ifs.read((char*)(mTemp.data), mTemp.elemSize() * mTemp.total());
		vecPC[i].mR = mTemp.clone();
		
		ifs.read((char*)(&type), sizeof(int));
		mTemp.create(rows, cols, type);
		ifs.read((char*)(mTemp.data), mTemp.elemSize() * mTemp.total());
		vecPC[i].mT = mTemp.clone();

	}

	ifs.close();

	//// create keyframe data for all of the loaded cloud frames
	//for (int i = 0; i < vecPC.size(); i++)
	//{
	//	Keyframe* pKeyframe = new Keyframe( &vecPC[i] );
	//	m_keyFrames.add( pKeyframe );
	//}

	m_nTrackingMode = RELOCALIZING;
}

void CReconstruction3D::relocalizePose()
{
	// we need to compute descriptors for the current frame, so we can compare to our loaded cloud keys

	trackToEstimatePose();  // we don't really need to track, but we need most of it to get the correct descriptors
	addDescriptorsToPC();

	//Keyframe* pKeyframe = new Keyframe( &algParams.vecPointCloud.back() );
	//
	//m_keyFrames.Relocalize( pKeyframe );

	// This might be easier in the other branch where we don't use optical flow for the first pass identification
}


