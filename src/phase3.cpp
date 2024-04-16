#include "iostream"
#include "fstream"
#include <string>
#include <time.h>
#include <math.h>
#include "Reconstruction3D.h"

#define iteration 10

using namespace std;
using namespace cv;

void readInfo(Mat& intrinsic, Mat& distcoeff, int& nStatus);


void main( int argc, char **argv )
{
	CReconstruction3D *pReconstruct3D = new CReconstruction3D();

	// Let the Reconstruction class parse the command line options
	pReconstruct3D->ParseCommandLineArguments( argc, argv );

	//global variables
	pcl::PointCloud<pcl::PointXYZ>::Ptr basicPointCloud;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	VideoCapture vc;
	AlgParams& algParams = pReconstruct3D->algParams;
	vector<pcl::PolygonMesh> vecTriangulates;
	Mat frame;
	int StartPoint = 0; //For Sync reasons, change the value w.r.t input video starting frame
	double Real_distance;
	double scale_factor = 597.55;
	int cnt = 0;
	int nPairCnt = 0;
	Mat canvas;
	int iter = 0;
	int64 tStart, tEnd;
	float pTime;                             
	float t_min = 0, t_max = 0;
	float t_ave = 0;
	float total_error = 0;

	// Load the descripted PC if instructed to load
	//vector<DescriptedPC> testLoadedPC;
	if (pReconstruct3D->GetBLoadCloud())
	{
		pReconstruct3D->LoadDescriptedPCVector( algParams.vecPointCloud, pReconstruct3D->GetStrLoadCloudName() );
	}

	//vc.open("test/data and output video/10sec/10sec_s7.mp4");
	//vc.open("ADASONE_Lens/INPUT.mp4");
	//WIDE-ANGLE
	//vc.open("Tested_WideAngle/Backward_motion/TV_CAM_device_20181120_084656_LS2_547-1640.avi");
	//vc.open("Ff9 angle mix.mp4");
	//vc.open("WA_DB_20181126/rec/rec_cuts/Direct linear increase [errorless]/8_58-238_undistort.avi");
	//vc.open("E:/WA_BD/20181220_NewDB/Wide_26.5_1_undistort.avi");
	//vc.open("E:/WA_BD/20181215_Y-direction dominant DB/3.avi");
	//pReconstruct3D->readInfo("Tested_WideAngle/W_IntrinsicMatrix.yml", "Tested_WideAngle/W_DistortionCoeffs.yml");
	//FISHEYE
	//vc.open("Tested_FishEye/FTV_CAM_device_20181120_094525lsbuild147-749.avi");
	pReconstruct3D->readInfo("Tested_FishEye/F_IntrinsicMatrix.yml", "Tested_FishEye/F_DistortionCoeffs.yml");
	//ADASONE
	//vc.open("ADASONE_Lens/rear1.avi");
	//pReconstruct3D->readInfo("ADASONE_Lens/IntrinsicMatrix_120.yml", "ADASONE_Lens/DistortionCoeffs_120.yml");
	//vc.open("WA_new/Wide-Angle/wcam1 (1).avi");
	//pReconstruct3D->readInfo("WA_new/Wide-Angle/IntrinsicMatrix.yml", "WA_new/Wide-Angle/DistortionCoeffs.yml");
	//vc.open("WA_Lens/videocuts/fcam1.avi");
	//vc.open("SelfCalib/W1.mp4");
	//pReconstruct3D->readInfo("SelfCalib/F11params/IntrinsicMatrix.yml", "SelfCalib/F11params/DistortionCoeffs.yml");
	//pReconstruct3D->readInfo("WA_Lens/fisheye/fisheye_parameter_files/fisheye_camParams5//IntrinsicMatrix.yml", "WA_Lens/fisheye/fisheye_parameter_files/fisheye_camParams5//DistortionCoeffs.yml");
	//pReconstruct3D->readInfo("WA_Lens/wide/wideangle_parameter_files/wide_camParams4///IntrinsicMatrix.yml", "WA_Lens/wide/wideangle_parameter_files/wide_camParams4///DistortionCoeffs.yml");
	//pReconstruct3D->readInfo("S6_cam//IntrinsicMatrix.yml", "S6_cam//DistortionCoeffs.yml");
	//pReconstruct3D->readInfo("S7_cam//IntrinsicMatrix.yml", "S7_cam//DistortionCoeffs.yml");
	/*Raspi-cam*/
	//pReconstruct3D->readInfo("Raspi_cam//IntrinsicMatrix.yml", "Raspi_cam//DistortionCoeffs.yml");
	vc.open("paperslams/f9_undistort.avi");
	//vc.open("E:/WA_BD/[PAPER] sensor paper evaluation methods/Figures for sensors/Experimental Data/Effect of distortion on vSfm and OD/lab_floor1_undistort.avi");
	//pReconstruct3D->readInfo("S5_cam//IntrinsicMatrix.yml", "S5_cam//DistortionCoeffs.yml");
	
	if (!vc.isOpened()){
		cout << "Failed to load the video" << endl;
		return;
	}
	string videooutname = "Phase2test.mpg";
	int nWidth =  (int)vc.get(CV_CAP_PROP_FRAME_WIDTH);
	int nHeight = (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT);
	Size framesize((int)vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT));
	VideoWriter wri(videooutname, CV_FOURCC('M', 'P', 'E', 'G'), 30.0, framesize);
	// Initialization of the structure
	pReconstruct3D->initParams(nWidth, nHeight);

	// Load meta data
	if (algParams.nStatus != NO_ERR){
		cout << "Input data files not found";
		return;
	}

	// Skip a number of frames at the beginning
	vc.set(CV_CAP_PROP_POS_FRAMES, 0);

	for (iter = 0; iter<iteration; iter++)
	{
		printf("iteration number : %d. ", iter + 1);

		tStart = cvGetTickCount();

	bool bDetectedCorners = false;	// Variable to indicate the detection of corner points is successful or not

	while (!bDetectedCorners){
		Mat img1;
		vc >> img1;
		if (img1.empty())
			return;
		cnt++;
		cout << "frame number: " << StartPoint+cnt << endl;
		bDetectedCorners = pReconstruct3D->findFirstSeedFrame(img1);
	}
	printf("First frame: %d\n", StartPoint + cnt);

	int base = 0;
	for (; !algParams.bIsKeyFrame; cnt++){
		vc >> algParams.img2;

		if (algParams.img2.empty())
			return;

		pReconstruct3D->detectCorrespondence(algParams.movingParams.nWinSize, algParams.movingParams.fErr, true);

		cout << "frame number: " << StartPoint + cnt << endl;

		pReconstruct3D->selectKeyFrame();

		if (algParams.bIsKeyFrame){
			pReconstruct3D->constructFirstPointCloud();
			if (pReconstruct3D->algParams.nStatus == ERR_FIND_CAM_MATRIX)
				algParams.bIsKeyFrame = false;
		}

		cvWaitKey(1);
	}

	//pReconstruct3D->geoEstimation(algParams.vecPointCloud[0].vecPoint3D);
	//pReconstruct3D->viewGeometry2D();
	//pReconstruct3D->modifyGeo();
	//return;
	//create a new point cloud and rebase to it


	pReconstruct3D->visualizePointCloud(algParams.vecPointCloud, pointCloud);
	pReconstruct3D->generateVirtualObject();//This _seems_ clean
	
	pReconstruct3D->init4Tracker();
	Point3d ptCamPose(0, 0, 0);
	algParams.vecRegisteredPoints3D.resize(3);
	algParams.vecRegisteredPoints3D[2].push_back(ptCamPose);

	//pReconstruct3D->visualizePointCloud(algParams.vecRegisteredPoints3D, pointCloud);
	// Track the camera and draw the virtual objects
	// Save CamPose [position of camera in 3D space] & Rotations
	ofstream out_data("translation1.dat");
	ofstream out_data1("roll.dat");
	ofstream out_data2("pitch.dat");
	ofstream out_data3("yaw.dat");
	ofstream out_data4("total_rotations.dat");
	for (;;cnt++){//cnt < 148
		cout << "frame number: " << StartPoint + cnt << endl;
		cout << "R: " << algParams.R << endl;
		cout << "T: " << algParams.T << endl;
		cout << "Camera Pose: " << ptCamPose << endl;
		//cout << "Real_Distance Travelled (in Centimeters): " << Real_distance << endl;

		out_data << StartPoint + cnt << "\t" << ptCamPose << "\t" << Real_distance << "cm." << "\n";
		out_data1 << StartPoint + cnt << "\t\t\t" << algParams.R.at<double>(0)  << "\n";
		out_data2 << StartPoint + cnt << "\t\t\t" << algParams.R.at<double>(1)  << "\n";
		out_data3 << StartPoint + cnt << "\t\t\t" << algParams.R.at<double>(2) << "\n";
		out_data4 << StartPoint + cnt << "\t" << algParams.R << "\n";
	
		out_data.flush();
		out_data1.flush();
		out_data2.flush();
		out_data3.flush();
		out_data4.flush();

		int t0 = clock();
		vc >> algParams.img2;
	
		if (algParams.img2.empty())
			break;

		if (pReconstruct3D->m_nTrackingMode == CReconstruction3D::TRACKING)
		{
			pReconstruct3D->trackToEstimatePose();//find corresponding points

			// add the camera translation only (algParams.T) to algParams.vecRegisteredPoints3D
			Mat R = algParams.R, T = algParams.T;
			//R = R.t();
			//T = -T.t()*R;
			//T = T.t();
			ptCamPose = Point3d(T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0));
			Real_distance = scale_factor * sqrt(pow(T.at<double>(0, 0), 2) + pow(T.at<double>(1, 0), 2) + pow(T.at<double>(2, 0), 2));
			algParams.vecRegisteredPoints3D[2].push_back(ptCamPose);
			//pReconstruct3D->visualizePointCloud(algParams.vecRegisteredPoints3D, pointCloud);
		}
		else //if (pReconstruct3D->m_nTrackingMode == CReconstruction3D::RELOCALIZING)
		{
			pReconstruct3D->relocalizePose();
		}

		wri.write(pReconstruct3D->insertVirtualObjects());//draw the objects using algParams.T and algParams.R

		if (algParams.bIsKeyFrame){
			pReconstruct3D->rebase();
			//also update the pose of the camera
			//the object sadly doesn't seem to be drawing
			//pReconstruct3D->visualizePointCloud(algParams.vecPointCloud[algParams.vecPointCloud.size()-1].vecPoint3D, pointCloud,algParams.nStatus);
			pReconstruct3D->visualizePointCloud(algParams.vecRegisteredPoints3D, pointCloud);

		}

		
		//pReconstruct3D->visualizePointCloud(algParams.vecRegisteredPoints3D, pointCloud);

		//cvWaitKey(0);

		tEnd = cvGetTickCount();
		int t = clock() - t0;
		pTime = 0.001*(tEnd - tStart) / cvGetTickFrequency();
		t_ave += pTime;

		//printf("processing time : %d ms\n", t);
		if (iter == 0)
		{
			t_min = pTime;
			t_max = pTime;
		}
		else
		{
			if (pTime<t_min) t_min = pTime;
			if (pTime>t_max) t_max = pTime;
		}
	}

	if (iteration == 1) t_ave = t_ave;
	else if (iteration == 2) t_ave = t_ave / 2;
	else t_ave = (t_ave - t_min - t_max) / (iteration - 2);

	//printf("\nAverage processing time : %.f ms\n", t_ave);

	}

	pReconstruct3D->visualizePointCloud(algParams.vecPointCloud, pointCloud, true);

	//Save the descripted PC to file
	if (pReconstruct3D->GetBSaveCloud())
	{
		pReconstruct3D->SaveDescriptedPCVector( algParams.vecPointCloud, pReconstruct3D->GetStrSaveCloudName() );
	}

	delete pReconstruct3D;
}