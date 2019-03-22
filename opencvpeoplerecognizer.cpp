#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "opencvpeoplerecognizer.h"

#include <QFileDialog>
#include <QVideoWidget>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::dnn;

using namespace cv;
using namespace dnn;

OpenCVPeopleRecognizer::OpenCVPeopleRecognizer(QString fileName){
        _fileName = fileName;
    };

void OpenCVPeopleRecognizer::analyzeVideo(){

    QMediaPlayer* player = new QMediaPlayer;
    QVideoWidget* vw = new QVideoWidget;

    player->setVideoOutput(vw);
    player->setMedia(QUrl::fromUserInput(_fileName));
    vw->setGeometry(100,100,300,400);
    vw->show();

    player->play();

    VideoCapture cap(_fileName.toUtf8().constData());
    Mat current_frame;

        /// Set up the pedestrian detector --> let us take the default one
    HOGDescriptor hog;
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

        /// Set up tracking vector
        vector<Point> track;

        while(true){
            /// Grab a single frame from the video
            cap >> current_frame;

            /// Check if the frame has content
            if(current_frame.empty()){
                cerr << "Video has ended or bad frame was read. Quitting." << endl;
            }

            Mat img = current_frame.clone();
            vector<Rect> found;
            vector<double> weights;

            hog.detectMultiScale(img, found, weights);

            /// draw detections and store location
            for( size_t i = 0; i < found.size(); i++ )
            {
                Rect r = found[i];
                rectangle(img, found[i], cv::Scalar(0,0,255), 3);
                stringstream temp;
                temp << weights[i];
                track.push_back(Point(found[i].x+found[i].width/2,found[i].y+found[i].height/2));
            }

            /// Show
            imshow("detected person", img);
            waitKey(1);
        }
};
