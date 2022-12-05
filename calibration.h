//
// Created by magnus on 11/30/22.
//

#ifndef PHOTOMETRIC_CALIBRATION_CALIBRATION_H
#define PHOTOMETRIC_CALIBRATION_CALIBRATION_H

void rectificationMap() {
    //fill matrices
    cv::Mat cam(3, 3, cv::DataType<float>::type);
    cam.at<float>(0, 0) = 1288.60f;
    cam.at<float>(0, 1) = 0.0f;
    cam.at<float>(0, 2) = 976.46228f;

    cam.at<float>(1, 0) = 0.0f;
    cam.at<float>(1, 1) = 1288.6499f;
    cam.at<float>(1, 2) = 575.25958f;

    cam.at<float>(2, 0) = 0.0f;
    cam.at<float>(2, 1) = 0.0f;
    cam.at<float>(2, 2) = 1.0f;

    //-0.12584973871707916,   -0.33098730444908142,    0.00008958806574810,   -0.00007964076212374,   -0.02069440484046936,    0.30147391557693481,   -0.48251944780349731,   -0.11008762568235397
    cv::Mat dist(8, 1, cv::DataType<float>::type);
    dist.at<float>(0, 0) = -0.1258497387f;
    dist.at<float>(1, 0) = -0.33098730444945f;
    dist.at<float>(2, 0) = 0.0000895880657481f;
    dist.at<float>(3, 0) = -0.00007964076;
    dist.at<float>(4, 0) = -0.0206944048404f;
    dist.at<float>(5, 0) = 0.3014739155f;
    dist.at<float>(6, 0) = -0.482519447803f;
    dist.at<float>(7, 0) = -0.110087625682353f;

    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(cam, dist, cv::Mat(), cam, cv::Size(1920, 1200), CV_32FC1, map1, map2);

    //cv::remap(srcImg, *m_undistImg, map1, map2, cv::INTER_CUBIC);
}

void distortCoordinates(float *x_, float *y_) {
    float cx = 976.46228f;
    float cy = 575.25958f;
    float fx = 1288.60f;
    float fy = 1288.6499f;

    float k1 = -0.1258497387f;
    float k2 = -0.33098730444945f;
    float p1 = 0.0000895880657481f;
    float p2 = -0.00007964076;
    float k3 = 0.3014739155f;;
    // Relative coordinates
    float x = (*x_ - cx) / fx;
    float y = (*y_ - cy) / fy;

    float r2 = x * x + y * y;

    // Radial distortion
    float xDistort = x * (1.0f + (k1 * powf(r2, 2)) + (k2 * powf(r2, 4)) + (k3 * powf(r2, 6)));
    float yDistort = y * (1.0f + (k1 * powf(r2, 2)) + (k2 * powf(r2, 4)) + (k3 * powf(r2, 6)));

    // Tangential distortion
    xDistort = xDistort + ((2 * p1 * x * y) + (p2 * (r2 + (2 * powf(x, 2)))));
    yDistort = yDistort + (p1 * (r2 * r2 + (2 * y * y)) + (2 * p2 * x * y));

    // Back to absolute coordinates.
    xDistort = xDistort * fx + cx;
    yDistort = yDistort * fy + cy;

    *x_ = xDistort;
    *y_ = yDistort;
}

#endif //PHOTOMETRIC_CALIBRATION_CALIBRATION_H
