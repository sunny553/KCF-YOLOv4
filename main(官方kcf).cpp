#include "yolo.h"
#include "track.h"

using namespace cv;

int main(int argc, char* argv[])
{
	Net_config yolov4_tiny = { 0.3, 0.2, 320, 320,"./cfg/coco.names",
		"./cfg/yolov4-tiny.cfg",
		"./cfg/yolov4-tiny.weights", "yolov4 - tiny" };
	YOLO model(yolov4_tiny);
	
	string imgpath = argv[1];
	VideoCapture cap(imgpath);
	Mat firframe,frame;	
	cv::Rect roi, tracking_rect,last_rect;

	while (1) {
		cap >> firframe;
		if (!firframe.data)
		{
			return 0;
		}	
		roi = model.detect_roi(firframe);
		if (roi.width != 0) {
			break;
		}
	}
	
	Ptr<TrackerKCF> tracker = TrackerKCF::create();
	tracker->init(firframe, roi);
	bool isfound = tracker->update(firframe, roi);
	last_rect = roi;
	namedWindow("tracker", WINDOW_AUTOSIZE | WINDOW_KEEPRATIO);
	for (;;)
	{
		cap >> frame;
		if (!frame.data)
		{
			break;
		}

		bool isfound = tracker->update(frame,tracking_rect);
		if (!isfound)
		{
			cout << "Yolo is used Looking for targets...\n";
			while (1) {				
				if (!frame.data)
				{
					return 0;
				}
				roi = model.detect_optimal_roi(frame,last_rect);
				rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);
				imshow("tracker", frame);
				waitKey(1);
				if (roi.width != 0) {
					break;
				}
				cap >> frame;
			}
			tracker->~TrackerKCF();
			tracker = TrackerKCF::create();
			tracker->init(frame, roi);
			isfound = tracker->update(frame, roi);
		}
		else{
			last_rect =tracking_rect;
			rectangle(frame, tracking_rect, Scalar(255, 0, 0), 2, 1);
			imshow("tracker", frame);
			waitKey(1);
		}
	}
	destroyAllWindows();
	return 0;
}


