#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "opencvpeoplerecognizer.h"
#include "yolorecognizer.h"

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>


#include <QFileDialog>
#include <QMediaPlayer>
#include <QVideoWidget>

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::dnn;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Video"), "/home/", tr("Video Files (*.mp4)"));
    OpenCVPeopleRecognizer recognizer= OpenCVPeopleRecognizer(fileName);
    recognizer.analyzeVideo();
}

void MainWindow::on_pushButton_2_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Video"), "/home/", tr("Video Files (*.mp4)"));

    YoloRecognizer recognizer = YoloRecognizer(fileName);
    recognizer.analyzeVideo();
}
