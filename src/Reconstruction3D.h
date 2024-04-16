#pragma once
//#ifdef _EXPORTS
//#define  __declspec(dllexport) 
//#else
//#define  __declspec(dllimport) 
//#endif


#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
//PCL
#include "pcl/io/ply_io.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/features/normal_3d.h"
#include "boost/thread/thread.hpp"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
//#include <pcl/registration/gicp6d.h>
//#include <pcl/registration/icp_nl.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac.h>
//#include <pcl/sample_consensus/sac_model_plane.h>
//#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include "boost/thread/thread.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "Keyframes.h"
#include "DescriptedPC.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace pcl;

struct TrackParams
{
	int nPoints = 500;								///< Number of maximum tracking points
	double dQuality = 0.01;							///< Quality of the corner 
	double dMinDistance = 12;						///< Minimum distance between 2 points to be tracked
	int nBlSize = 5;								///< Block size of neighborhood used to compute the score for corner quality
	int nWinSize = 21;								///< Size of window to search for the tracking point in the second image 
	float fErr = 15;								///< Error threshold to evaluate a point is found/not-found
};

struct VirtualObject{
	vector<Point3f> vertices3D;
	vector<Point2f> vertices2D;
	vector<vector<int>> faces;
	Point2f currentLocCen;		// Current location center of the object
};

struct SceneBoundingBox{
	double dMinX;
	double dMaxX;
	double dMinY;
	double dMaxY;
	double dMinZ;
	double dMaxZ;
};

enum ERR_CODE{
	NO_ERR,
	ERR_NO_CAM_INTRINSIC,
	ERR_NO_CAM_DISTORTION,
	ERR_FIND_CAM_MATRIX,
	ERR_REBASE
};

struct AlgParams
	///< This struct holds the important inputs/outputs of the program
{
	//vector<vector<Point3d>> vecPoints3D;			///< Contains the originally reconstructed 3D points of each view-pair (size of outer vector indicates # view pairs)
	vector<DescriptedPC>	vecPointCloud;
	vector<vector<Point3d>> vecRegisteredPoints3D;	///< Contains the reconstructed 3D points of each view-pair after registration (size of outer vector indicates # view pairs) - isn't this redundant with vecPointCloud.vecPoint3D above?
	Mat intrinsic;									///< Contains the camera intrinsic matrix (loaded from file - need to calibrate the cam beforehand)
	Mat distcoeff;									///< Contains the camera distortion matrix (loaded from file - need to calibrate the cam beforehand)
	Mat img1;										///< Contains the first image of each view-pair
	Mat img2;										///< Contains the second image of each view-pair
	vector<VirtualObject> vVirObjects;				///< The virtual object to insert to the scene
	Mat P;											///< The camera matrix of the first view //Projection Matrix
	Mat P1;											///< The camera matrix of the second view //Projection Matrix
	Mat T;											///< Translation vector/matrix of the current camera pose in the defined coordinate
	Mat R;											///< Rotation matrix of the current camera pose in the defined coordinate
	Mat R_KF;
	Mat T_KF;
	TrackParams		movingParams;					///< Params for KLT tracking when camera moving
	int				nStatus = 0;					///< Program runtime error status
	bool			bIsKeyFrame = false;			///< Flag to check if the current frame is a key frame for a new point cloud reconstruction
};

class CReconstruction3D
{
public:

	AlgParams algParams;
	 CReconstruction3D();
	 ~CReconstruction3D();
	 void initParams(int nWidth, int nHeight);
	///< Initialize the parameters before running the program0
	 bool detectSeedPoints(int nPtMax, double qualityLevel, double minDistance, int blockSize, bool bFirst = false);
	///< Detect the good points (corners) for tracking
	///< Inputs: parameters for KLT optical flow algorithm
	 void detectCorrespondence(int nWinSize, float er, bool showMatchedPoints = false);
	///<  Find the corresponding points by tracking from 2 images
	///< Inputs:
	///< corner1: the key points in the first image to find the coresspondence
	///< corner2: the detected points of the key points on the second image
	///< nWinSize: size of window to search the point for
	///< er: error threshold to evaluate if a point is accurately found
	///< showMatchedPoints: a flag to indicate if to show the reslt of detection
	 void readInfo(string intrinsicPath, string coeffPath);

	 void reconstructFrom2Views();
	///< 3D point reconstruction
	 bool findFirstSeedFrame(Mat & image);
	 void visualizePointCloud(vector<vector<Point3d>>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, bool bStop2View = true);
	 void visualizePointCloud(vector<DescriptedPC>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, bool bStop2View = true);
	 void visualizePointCloud(vector<Point3d>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, bool bStop2View = true);
	
    // void iterativeClosestPoint(, float maxCorrespondence, int maxIterations, float transformationEpsilon, float euclideanFitnessEspilon, float scale);
	///< Convert the 3D points from vector type to PCL type for displaying in PCL platform
	 
	///< Generate tris and verts from clustered point clouds by Delaunay's triangulation method using PCL library (not valid in android version yet)
	 Mat insertVirtualObjects();

	 void saveGeometry(vector<pcl::PolygonMesh>& vecTriangulates);
	 void selectKeyFrame();
	 void selectKeyFrame(vector<Point2f> & imgPoints1, vector<Point2f> & imgPoints2, vector<uchar>& mask);

	 void generateVirtualObject();
	 void trackToEstimatePose();
	 void rebase();
	 void constructPointCloudInReverse();
	 void relocalizePose();
	 
	 void init4Tracker();
	 void constructFirstPointCloud();

	 void geoEstimation(vector<Point3d>& pointCloud);
	 void geoEstimation(vector<vector<Point3d>>& pointCloud);
	 void viewGeometry2D();

	 void modifyGeo();

	 void SaveDescriptedPCVector(vector<DescriptedPC>& vecPC, string sFilename);
	 void LoadDescriptedPCVector(vector<DescriptedPC>& vecPC, string sFilename);

	 enum { TRACKING, RELOCALIZING } m_nTrackingMode = TRACKING;

	 void ParseCommandLineArguments( int argc, char **argv );

	 bool GetBLoadCloud() { return m_bLoadCloud; }
	 bool GetBSaveCloud() { return m_bSaveCloud; }
	 string	GetStrLoadCloudName() { return m_strLoadCloudName; }
	 string	GetStrSaveCloudName() { return m_strSaveCloudName; }

protected:
	void addDescriptorsToPC();

	Mat calculateProjectionMatrix(vector<Point3d>  prevPoints3D, vector<Point2f>  nextPoints);
	bool detectSeedPointsForImage(Mat & grayImg, int nPtMax, double qualityLevel, double minDistance, int blockSize, bool bFirst = false);
	 float reprojectionError(vector<int> & inliers, Mat & rotationVector, Mat & translationVector, vector<Point3d> & points3D, vector<Point2d> & points2D);
	 vector<int> PnPRansac(vector<Point3d>  & prevPoints3D, vector<Point2d> & nextPoints);
	 void rejectPoint3D(vector<Point2d>& vimgPointsInlier1, vector<Point2d>& vimgPointsInlier2, vector<Point3d> &vecReconstruction3D);
	 Mat estimateRotation(Mat & First, Mat & Second);

	 void drawMatchingPoints(Mat& canvas, vector<Point2d>& imgPoints1, vector<Point2d>& imgPoints2, int nMode = 0);
	 void drawMatchingPoints(Mat& img1, Mat& img2, Mat& canvas, vector<Point2f>& imgPoints1, vector<Point2f>& imgPoints2, int nMode = 0);
	 void drawMatchingPoints(Mat& canvas, vector<Point2f>& imgPoints1, vector<Point2f>& imgPoints2, int nMode = 0);
	 void drawMatchingPoints(Mat& canvas, vector<Point2d>& imgPoints1, vector<Point2f>& imgPoints2, int nMode = 0);
	///< Display the point cloud in PCL platform
	 void createPointCloud(vector<Point3d>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZ>::Ptr &pointCloud);
	 void createPointCloud(vector<vector<Point3d>>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZ>::Ptr &pointCloud);
	 void createPointCloud(vector<Point3d>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, vector<Scalar> &colors);
	 void createPointCloudv(vector<vector<Point3d>>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, vector<Scalar> &colors);
	 void createPointCloudv(vector<DescriptedPC>& vecReconstruction3D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud, vector<Scalar> &colors);
	 void distanceVector2D(vector<Point2d>& imgPoints1, vector<Point2d>& imgPoints2, double& dMean, double& dStd);
	 void distanceVector2D(vector<Point2f>& imgPoints1, vector<Point2f>& imgPoints2, double& dMean, double& dStd, vector<uchar>& mask);

	 ///< Find the mean distance and std. devia from element-wise of two vectors of 2D points
	///< Inputs: 
	///< imgPoints1: the first list of 2D points
	///< imgPoints2: the second list of 2D points of the same size
	///< dMean: mean distance between the corresponding points
	///< dStd: the standard deviation of the distances

	 void rejectOutlier(const vector<Point2f>& imagePoints1, const vector<Point2f>& imagePoints2, vector<Point2f>& imagePointsInlier1, vector<Point2f>& imagePointsInlier2, float minDist, float qualityRatio, bool bUpdateOFStt = true);
	 void rejectOutlier(const vector<Point2d>& imagePoints1, const vector<Point2d>& imagePoints2, vector<Point2d>& imagePointsInlier1, vector<Point2d>& imagePointsInlier2, float minDist, float qualityRatio, bool bUpdateOFStt = true);
	 void rejectOutlier(const vector<Point2f>& imagePoints1, const vector<Point2f>& imagePoints2, vector<uchar>& imagePointsInlier, float minDist, float qualityRatio, bool bUpdateOFStt = true);
	 void rejectOutlier(const vector<Point2d>& imagePoints1, const vector<Point2f>& imagePoints2, vector<uchar>& imagePointsInlier, float minDist, float qualityRatio, bool bUpdateOFStt = true);
	///< Reject the wrongly tracked points by epipolar line property

	 void cvtMat2Vec(Mat& minput, vector<Point2d>& voutput);
	///< Convert from Mat type to vector type

	 void cvtMat2Vec(Mat& minput, vector<Point3d>& voutput);
	 void cvtMat2Vec(Mat& minput, vector<Point3f>& voutput);
	///< Convert from Mat type to vector type

	 void cvtMat32F2Vec3D(Mat& input, vector<Point3d>& output);
	///< Convert from Mat type to vector type

	 void cvtVec2Mat(vector<Point3d>& vinput, Mat& moutput);
	///< Convert from Vector type to Mat type

	 void cvtVec2Mat(vector<Point2d>& vinput, Mat& moutput);
	///< Convert from Vector type to Mat type

	 void cvtVec2Mat(vector<Point3f>& vinput, Mat& moutput);
	///< Convert from Vector type to Mat type

	 void mySVD(Mat& input, Mat& U, Mat& S, Mat& V);

	 void findCameraMatrix(vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2, Mat& intrinsic,
		 Mat& distcoeff, Mat& P, Mat& P1, Mat& F, int& nStatus);

	 void estimateFundamentalMatrix();

	 void point3DReconstruct1(Mat& P, Mat& P1, Mat &intrinsic, Mat &distcoeff, vector<Point2d> vimgPointsInlier1, vector<Point2d> vimgPointsInlier2, vector<Point3d> &vecReconstruction3D);

	 void updateCorrespondingPoint(Mat& P, Mat& P1, Mat &intrinsic, Mat &distcoeff, vector<Point2d>& vimgPointsInlier1, vector<Point2d>& vimgPointsInlier2, double thres, double &avgErr, vector<double> &verr);

	 void reconstruct3DPoints(vector<Point3d>& vecReconstruction3D);

	 void findCameraMatrixIter(vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2, Mat& intrinsic,
		 Mat& distcoeff, Mat& P, Mat& P1, Mat& F, int& nStatus);
	 ///< Find camera matrix iteratively by reducing the wrong pair of corresponding points

	 void reprojectionErr(vector<Point3d> &vecReconstruction3D, Mat& R, Mat& T, Mat &intrinsic, Mat &distcoeff, vector<Point2d> &imgPts, double &avgErr, vector<double> &verr);

	 Mat camMatrix(Mat& R, Mat& T);

	 Mat camMatrix(Mat& R, Mat& T, Mat& intrinsic);

	 void chooseRealizableSolution(vector<Mat>& Rs, vector<Mat>& Ts, vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2, Mat& intrinsic, Mat& R, Mat& T);
	///< Select the right rotation/translation matrices from possible ones after finding fundamental matrix

	 Mat triangulateMidPoint(vector<Point2d>& imagePoints1, vector<Point2d>& imagePoint2, Mat P, Mat P1);
	///< Used inside chooseRealizableSolution

	 void point3DReconstruct(Mat& P, Mat& P1, vector<Point2d>& vimgPointsInlier1, vector<Point2d>& vimgPointsInlier2, Mat& matReconstruct3D);
	///< Reconstructing 3D points from the calculated camera matrices and corresponding point

	 bool checkCoherentRotation(Mat& R);

	 void createColors(vector<Scalar>& colors);
	///< Create look-up-table color palette for display
	 bool detectSeedPoints4PCloud(int nPtMax, double qualityLevel, double minDistance, int blockSize, bool bFirst = false);

	 double ED(const vector<double> &point_a, const vector<double> &point_b);
	 double ED(const Point2d &point_a, const Point2d &point_b);
	 double ED(const Point3d &point_a, const Point3d &point_b);
	///< Compute the Euclidean distance between 2 points

	 boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis(
		pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals);

	 double triangulatePoints1(const vector<Point2d>& vImgPoints1, const vector<Point2d>& vImgPoints2, const Mat& intrinsic, Mat& P, Mat& P1, vector<Point3d>& pointClound);

	 Mat linearLSTriangulation(Point3d& u, Mat& P, Point3d& u1, Mat& P1);
	///< Find 3D coordinates by triangulating algorithm

	 void removeAbnormalPoints3D(vector<Point3d>& vecInput, vector<Point3d>& vecOutput);

	 double estimateMSBandwidth(const Mat& mData3D);
	///< Estimate the bandwidth for meanshift clustering using Silverman method, based on the standard devia. 

	
	//void set_kernel(double(*_kernel_func)(double, double));
	 vector<double> shift_point(const vector<double> &, const vector<vector<double> > &, double);

	vector<vector<double> > MSCluster(vector<vector<double> >, double);
	///< Meanshift clustering

	double(*kernel_func)(double, double);

	 float ComputeDFScore(const Mat& mInput8UC1);
	 float ComputeDFScore2(const Mat& mInput8UC1);

	 void HaarWavelet(Mat &src, Mat &dst, int NIter);
	 
	 bool rotationCheck(vector<Point2f> m_vecInliners1F, vector<Point2f> m_vecInliners2F);

	 void estimateSceneBBox();

	 void estimateCamPose();

	 void drawVirObjectOnImage(Mat& image, VirtualObject& vObj, int nPolySize = 4);

	 void verifyNewPC(vector<Point3d>& vecReconstruct3D, double dThresh);

	 float ComputeAngle(Point2f p1, Point2f p2);

	// Class variables
	SceneBoundingBox m_sBoundingBox3D;
	Mat				m_mDisplay;
	Mat				m_mCanvas;
	vector<Point2d> m_vImgPointsInlier1;
	vector<Point2d> m_vImgPointsInlier2;
	vector<Point2d> m_vImgPointsInlier1_distort;
	vector<Point2d> m_vImgPointsInlier2_distort;
	Mat				m_mInlierMask;
	vector<Point2f> m_vecLastSeedFrame2D;				///< 2D image points of the last seed frame
	//vector<Point3d> m_vecLastSeedFrame3D;				///< 3D point cloud points from the last seedframe pair
	DescriptedPC m_sLastPointCloud;
	Mat m_mReconstruct3D;
	Mat m_mGrayImg1;									///< The gray-scale version of the first image
	Mat m_mGrayImg2;									///< The gray-scale version of the second image
	
	vector<Mat> m_vecPositionMatricies;					///< Key frame position matricies
	Mat m_mSeedFrameR;									///< rotation vector of the last seed frame
	Mat m_mPrevR;										///< Rotation vector of the last frame, used for extrapolation
	Mat m_mSeedFrameT;									///< translation vector of the last seed frame
	Mat m_mPrevT;										///< Previous translation vector (last frame), used for extrapolation purposes
	Mat m_mF;
	Mat m_mROI;
	vector<Mat>		m_vecCameraMatrices;
	vector<Point2f> m_vecImgSeedPoints;					///< Points to be tracked when the camera is moving						
	vector<Point2d> m_vecImgPoints1;						///< Points to be tracked from the first image						
	vector<Point2d> m_vecImgPoints2;						///< Corresponding points from the second image tracked from imgPoints1
	vector<Point2f> m_vecImgPoints1F;					///< Corresponding points from the second image tracked from imgPoints1
	vector<Point2f> m_vecImgPoints2F;					///< Corresponding points from the second image tracked from imgPoints1
	vector<Point2f> m_vecInliners1F;					///< Points to be tracked from the first image						
	vector<Point2f> m_vecInliners2F;					///< Points to be tracked from the first image						
	vector<uchar>	m_vecOFTrackStatus;					///< Flag vector to check the status of point tracking	
	vector<uchar>	m_vecInlinerStatus;					///< Flag vector to check the status of point tracking	

	vector<Scalar>	m_vecColors;							///< Colors pallette for display only
	int				m_nFramesFromBase;					///< number of frames from a seedframe

	TrackParams		m_sFundamentalParams;				///< Params for KLT tracking when finding fundamental matrix
	TrackParams		m_sReconstructParams;				///< Params for KLT tracking when fingding points for reconstructing 3D points

	// System thresholds
	int				m_nNofPairs = 1;					///< Number of view-pairs 
	double			m_dMeanThresh = 25.;				///< Mean distance to start a new view-pair.
	double			m_dStdThresh = 8.;					///< Standard deviation threshold to start a new view-pair ( to avoid parrallel views)
	float			m_fDefocusScoreThresh = 5.0f;		///< Threshold for the defocus score, the smaller the sharper
	//float			m_fDefocusScoreThresh = 7.f;		///< Threshold for the defocus score, the smaller the sharper
	double			m_d3DReprojectErrThresh = 50.;
	double			m_dRotationMaxThresh = 0.15;
	bool			m_bUseKalmanTracker = true;
	int				m_nMinSizeOfInliersThresh = 50;
	int				m_nSizeOfInliners;
	double			m_dDFAccum = 0;
	double			m_dFrameIntervalCnt = 0;

	Mat				m_mLastR;

	Point3f			m_prev3DPointsMass;
	Point3f			m_current3DPointsMass;


	int				m_nImgWidth;						///< width of the image input stream
	int				m_nImgHeight;						///< height of the image input stream
	int				m_nInliersCnt;						///< numer of inliers
	KalmanFilter    m_camPoseKF_R;
	KalmanFilter    m_camPoseKF_T;
	int				m_nCounter = 0;							///< Use to count the number of frame before using Kalman value for camera pose
	float			m_fInliersRate;
	// Feature description
	Ptr<Feature2D>				m_pDescDetector;
	Ptr<DescriptorExtractor>	m_pDescExtractor;
	Ptr<DescriptorMatcher>		m_pDescMatcher;
	Mat							m_mDescriptor1;
	Mat							m_mDescriptor2;
	vector<KeyPoint>			m_vecKeypoints1;
	vector<KeyPoint>			m_vecKeypoints2;
	vector<KeyPoint>			m_vecMatched1;
	vector<KeyPoint>			m_vecMatched2;
	vector<DMatch>				m_DescMatches;

	pcl::PolygonMesh			m_polyMesh;
	VirtualObject				m_delaunyGeo;

//	Keyframes                   m_keyFrames;                         // the keyframes database, used for matching using BoW
	bool						m_bLoadCloud;
	bool						m_bSaveCloud;
	string						m_strLoadCloudName;
	string						m_strSaveCloudName;
};

