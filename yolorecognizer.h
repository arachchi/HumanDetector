#ifndef YOLORECOGNIZER_H
#define YOLORECOGNIZER_H

#include <QFileDialog>
#include <QMediaPlayer>
#include <QVideoWidget>

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::dnn;

class YoloRecognizer
{
private:
    QString _fileName;
    String modelConfiguration = "/Users/personal/bin/tutorials/learnopencv/ObjectDetection-YOLO/yolov3.cfg";
    String modelWeights = "/Users/personal/bin/tutorials/learnopencv/ObjectDetection-YOLO/yolov3.weights";

    float CONFIDENT_THRESHOLD = 0.3; // Confidence threshold
    float NON_MAXIMUM_SUPPRESSION = 0.4;  // Non-maximum suppression threshold
    int INPUT_IMAGE_WIDTH = 416;  // Width of network's input image
    int INPUT_IMAGE_HEIGHT = 416; // Height of network's input image

    void postprocess(Mat& frame, const vector<Mat>& out);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
    vector<String> getOutputsNames(const Net& net);

public:
    YoloRecognizer(QString fileName);

    void analyzeVideo();
};

#endif // YOLORECOGNIZER_H
