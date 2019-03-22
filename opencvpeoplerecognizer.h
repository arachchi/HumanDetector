#ifndef OPENCVPEOPLERECOGNIZER_H
#define OPENCVPEOPLERECOGNIZER_H

#include <QMediaPlayer>

class OpenCVPeopleRecognizer
{
private:
    QString _fileName;

public:
    OpenCVPeopleRecognizer(QString fileName);

    void analyzeVideo();
};

#endif // OPENCVPEOPLERECOGNIZER_H
