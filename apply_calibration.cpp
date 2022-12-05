//
// Created by magnus on 12/5/22.
//

#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct VignetteMap {
    float *d;
    float *inv;
} vignetteMap;

void readVignetteMap() {
    std::string vignetteImage = "../result/vignetteCalibResult/vignetteSmoothed.png";
    printf("Reading Vignette Image from %s\n", vignetteImage.c_str());
    cv::Mat vignetteMat = cv::imread(vignetteImage, cv::IMREAD_UNCHANGED | cv::IMREAD_GRAYSCALE);
    uint32_t w = vignetteMat.cols;
    uint32_t h = vignetteMat.rows;
    vignetteMap.d = new float[w * h];
    vignetteMap.inv = new float[w * h];
    if (vignetteMat.rows != h || vignetteMat.cols != w) {
        printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d. Set vignette to 1.\n",
               vignetteMat.cols, vignetteMat.rows, w, h);
        return;
    }

    if (vignetteMat.type() == CV_8U) {
        float maxV = 0;
        for (int i = 0; i < w * h; i++)
            if (vignetteMat.at<unsigned char>(i) > maxV) maxV = vignetteMat.at<unsigned char>(i);

        for (int i = 0; i < w * h; i++)
            vignetteMap.d[i] = vignetteMat.at<unsigned char>(i) / maxV;
    } else if (vignetteMat.type() == CV_16U) {
        float maxV = 0;
        for (int i = 0; i < w * h; i++)
            if (vignetteMat.at<ushort>(i) > maxV) maxV = vignetteMat.at<ushort>(i);

        for (int i = 0; i < w * h; i++)
            vignetteMap.d[i] = vignetteMat.at<ushort>(i) / maxV;
    }

    for (int i = 0; i < w * h; i++)
        vignetteMap.inv[i] = 1.0f / vignetteMap.d[i];


    cv::Mat im_color;
    cv::Mat vignetteScaled(h, w, CV_32FC1, vignetteMap.d);
    vignetteScaled.convertTo(vignetteScaled, CV_8UC1, 255);
    cv::applyColorMap(vignetteScaled, im_color, cv::COLORMAP_HOT);

    cv::imwrite("../result/vignetteCalibResult/VignetteColored.png", im_color);
    cv::imshow("ColorMap", im_color);
    cv::waitKey(0);

    printf("Successfully read photometric calibration!\n");
}

int main() {


    float GInv[256];
    // load the inverse response function
    std::string pCalibFile = "../result/photoCalibResult/pcalib.txt";
    std::ifstream f(pCalibFile.c_str());
    std::string line;
    std::getline(f, line);
    std::istringstream l1i(line);
    std::vector<float> GInvvec = std::vector<float>(std::istream_iterator<float>(l1i), std::istream_iterator<float>());
    if (GInvvec.size() != 256) {
        printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected 256!\n",
               (int) GInvvec.size());
    }

    for (int i = 0; i < 256; i++)
        GInv[i] = GInvvec[i];

    float min = GInv[0];
    float max = GInv[255];
    for (float &i: GInv) i = 255.0f * (i - min) / (max - min);            // make it to 0..255 => 0..255.

    bool isGood = true;
    for (int i = 0; i < 255; i++) {
        if (GInv[i + 1] <= GInv[i]) {
            printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
            isGood = false;
        }
    }
    if (isGood)
        printf("Loaded camera inverse response function\n");

    readVignetteMap();


    std::string files = "../s30_vignette/";
    std::ifstream inFile, exFile;
    std::map<std::string, std::vector<float>> data;

    //lazycsv::parser parser{files + "times.csv"};
    std::vector<std::string> imageFileNames;
    std::vector<double> exposureTimes; // in ms
    imageFileNames.reserve(1200);
    exposureTimes.reserve(1200);

    for (const auto &entry: std::filesystem::directory_iterator("../s30_vignette/constant_exposure/images"))
        imageFileNames.emplace_back(entry.path().filename());

    std::string intrinsicsFile = (files + "intrinsics.yml");
    inFile.open(intrinsicsFile.c_str());
    if (!inFile) {
        fprintf(stderr, "failed to open '%s' for reading\n",
                intrinsicsFile.c_str());
    }
    std::vector<cv::Mat> images;

    images.reserve(1200);
    printf("Loading images\n");
    for (size_t i = 0; i < imageFileNames.size() / 2; ++i) {
        images.emplace_back(
                cv::imread(files + ("constant_exposure/images/" + imageFileNames[i]), cv::IMREAD_GRAYSCALE));
        if (images[i].data == NULL)
            throw std::runtime_error("Failed to load image: " + imageFileNames[i]);
    }
    printf("Loaded %zu images\n", images.size());
    mkdir("calibrationResult", 0777);

    uint32_t w = images[0].cols;
    uint32_t h = images[0].rows;

    for (int i = 0; i < images.size(); ++i) {
        auto *image = new float[w * h];
        int v = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                image[v] = (float) images[i].at<uchar>(y, x);
                v++;
            }
        }

        for (int i = 0; i < (w * h); i++) {
            image[i] = GInv[static_cast<unsigned char>(image[i])] * vignetteMap.inv[i];
        }

        cv::Mat dbgImg(h, w, CV_8UC3);
        for (int i = 0; i < (h * w); i++)
            dbgImg.at<cv::Vec3b>(i) = cv::Vec3b(image[i], image[i], image[i]);

        cv::imshow("Window", dbgImg);
        cv::imshow("Original", images[i]);
        delete[] image;
        cv::waitKey(0);
    }

}