#include "yolo.h"

YOLO::YOLO(Net_config config)
{
	cout << "Net use " << config.netname << endl;
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->inpWidth = config.inpWidth;
	this->inpHeight = config.inpHeight;
	strcpy_s(this->netname, config.netname.c_str());

	ifstream ifs(config.classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);

	this->net = readNetFromDarknet(config.modelConfiguration, config.modelWeights);
	this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
	this->net.setPreferableTarget(DNN_TARGET_CPU);
}

void YOLO::postprocess(Mat& frame, const vector<Mat>& outs,map<int, Rect> &flag)
// Remove the bounding boxes with low confidence using non-maxima suppression
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	//不同的模型的输出可能不一样，yolo的输出outs是[[[x,y,w,h,...],[],...[]]],
	//之所以多一维，是因为模型输入的frame是四维的，第一维表示帧数，如果只有一张图片推理，那就是1
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		//data是指针，每次从存储一个框的信息的地址跳到另一个框的地址
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			// 找到最大的score的索引，刚好对应80个种类的索引
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > this->confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame,i);
		flag[i]=box;
	}
}

void YOLO::drawPred(int classId, float conf,
	int left, int top, int right, int bottom, Mat& frame,int i)
	// Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!this->classes.empty())
	{
		CV_Assert(classId < (int)this->classes.size());
		label = this->classes[classId] + ":" + label;
	}
	label = to_string(i+1) + ":" + label;
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75,
		Scalar(0, 255, 0), 1);
}

void YOLO::detect(Mat& frame, map<int, Rect>& flag)
{
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0,
		Size(this->inpWidth, this->inpHeight),
		Scalar(0, 0, 0), true, false);

	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	this->postprocess(frame, outs,flag);

	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("%s Inference time : %.2f ms", this->netname, t);
	putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
	//imwrite(format("%s_out.jpg", this->netname), frame);
}

cv::Rect YOLO::detect_roi(cv::Mat& img) {
	map<int, Rect> flag;
	//cv::Mat copy_img;
	//img.copyTo(copy_img);
	detect(img,flag);
	static const string kWinName = "All Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_AUTOSIZE | WINDOW_KEEPRATIO);
	imshow(kWinName, img);
	waitKey(0);
	destroyWindow(kWinName);
	if (flag.size() == 0) {
		cout << "没有检测到目标" << endl;
		return cv::Rect(0, 0, 0, 0);
	}
	cout << "选择跟踪目标编号" << endl;
	int target_flag;
	cin >> target_flag;	
	/*rectangle(copy_img, flag[target_flag-1], cv::Scalar(255, 0, 0));
	namedWindow("KCF Initial ROI", WINDOW_AUTOSIZE | WINDOW_KEEPRATIO);
	imshow("KCF Initial ROI", copy_img);
	waitKey(0);	*/
	destroyAllWindows();

	return flag[target_flag-1];
}

cv::Rect YOLO::detect_optimal_roi(cv::Mat& img, cv::Rect last_roi) {
	map<int, Rect> flag;
	cv::Mat copy_img;
	img.copyTo(copy_img);
	cv::Rect best_fit_rect;
	detect(img, flag);
	if (flag.size() == 0) {
		return best_fit_rect;
	}
	float iou, best_iou;
	best_iou = -1;
	for (int i = 0; i < flag.size(); i++) {
		iou=DIoU(flag[i], last_roi);

		if (iou > -1 && iou > best_iou) {
			best_fit_rect = flag[i];
			best_iou = iou;
		}
	}
	copy_img.copyTo(img);
	return best_fit_rect;
}

float YOLO::DIoU(cv::Rect rect1, cv::Rect rect2) {
	cv::Rect rect_minimum_bounding_box = rect1 | rect2;
	cv::Rect rect_intersection = rect1 & rect2;
	float iou = float(rect_intersection.area()) / (float(rect1.area()) + float(rect2.area()) - float(rect_intersection.area()));   //简单交并比
	cv::Point2f center1, center2;
	center1.x = rect1.x + rect1.width / 2.0;
	center1.y = rect1.y + rect1.height / 2.0;
	center2.x = rect2.x + rect2.width / 2.0;
	center2.y = rect2.y + rect2.height / 2.0;
	float d = (center1.x - center2.x) * (center1.x - center2.x) + (center1.y - center2.y) * (center1.y - center2.y);
	float c = rect_minimum_bounding_box.width * rect_minimum_bounding_box.width + rect_minimum_bounding_box.height * rect_minimum_bounding_box.height;
	return iou - d / c;

}
