#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // 类别置信阈值
	float nmsThreshold;  // 非极大值抑制阈值
	int inpWidth;  // Width of network's input image
	int inpHeight; // Height of network's input image
	string classesFile;
	string modelConfiguration;
	string modelWeights;
	string netname;
};

class YOLO
{
public:
	YOLO(Net_config config);
	void detect(Mat& frame, map<int, Rect>& flag);	
	cv::Rect detect_roi(cv::Mat& img);
	cv::Rect detect_optimal_roi(cv::Mat& img,cv::Rect last_roi);
	
private:
	float confThreshold;
	float nmsThreshold;
	int inpWidth;
	int inpHeight;
	char netname[20];
	vector<string> classes;
	Net net;
	void postprocess(Mat& frame, const vector<Mat>& outs, map<int, Rect>& flag);
	void drawPred(int classId, float conf,
		int left, int top, int right, int bottom, Mat& frame,int i);
	float DIoU(cv::Rect yolo_rect, cv::Rect last_roi);
};



//Net_config yolo_nets[4] = {
//	{0.5, 0.4, 416, 416,"coco.names",
//	"yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3"},
//
//	{0.5, 0.4, 608, 608,"coco.names",
//	"yolov4/yolov4.cfg", "yolov4/yolov4.weights", "yolov4"},
//
//	{0.5, 0.4, 320, 320,"coco.names",
//	"yolo-fastest/yolo-fastest-xl.cfg",
//	"yolo-fastest/yolo-fastest-xl.weights", "yolo-fastest"},
//
//	{0.5, 0.4, 320, 320,"coco.names",
//	"yolobile/csdarknet53s-panet-spp.cfg",
//	"yolobile/yolobile.weights", "yolobile"}
//};
