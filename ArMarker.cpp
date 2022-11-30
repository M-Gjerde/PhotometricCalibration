//
// Created by magnus on 11/22/22.
//


#include <opencv2/core/mat.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>

int main(){

    cv::Mat markerImage;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::drawMarker(dictionary, 23, 200, markerImage, 1);
    cv::imwrite("marker23.png", markerImage);

    cv::Mat inputImage;

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 2);

    cap.set(cv::CAP_PROP_FPS, 30);

    if (!cap.isOpened())
    {
        std::cerr << "cannot open camera\n";
        return 1;
    }
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    std::vector<int> markerIds;

    while (true)

    {
        cap >> inputImage;
        cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        cv::aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);
        cv::imshow("Display window", inputImage);
        if (cv::waitKey(1) == 27)
            break;
    }



    return 0;
}