#include <iostream>
#include <opencv2/opencv.hpp>
#include "CalibrationYaml.hh"
#include "lazycsv.h"
#include <eigen3/Eigen/Core>

Eigen::Vector2d rmse(std::vector<double> G, std::vector<double> E, std::vector<double> &exposureVec,
                     const std::vector<cv::Mat> &dataVec, int wh) {
    long double e = 0;        // yeah - these will be sums of a LOT of values, so we need super high precision.
    long double num = 0;

    for (int i = 0; i < dataVec.size(); i++) {
        for (int k = 0; k < wh; k++) {
            if (dataVec[i].data[k] == 255) continue;
            double r = G[dataVec[i].data[k]] - exposureVec[i] * E[k];
            if (!std::isfinite(r)) continue;
            e += r * r * 1e-10;
            num++;
        }
    }

    return Eigen::Vector2d(1e5 * sqrtl((e / num)), (double) num);
}

void plotE(std::vector<double> E, int w, int h, std::string saveTo = "") {

    // try to find some good color scaling for plotting.
    double offset = 20;
    double min = 1e10, max = -1e10;

    double Emin = 1e10, Emax = -1e10;

    for (int i = 0; i < w * h; i++) {
        double le = log(E[i] + offset);
        if (le < min) min = le;
        if (le > max) max = le;

        if (E[i] < Emin) Emin = E[i];
        if (E[i] > Emax) Emax = E[i];
    }

    cv::Mat EImg = cv::Mat(h, w, CV_8UC3);
    cv::Mat EImg16 = cv::Mat(h, w, CV_16U);

    for (int i = 0; i < w * h; i++) {
        float val = 3 * (exp((log(E[i] + offset) - min) / (max - min)) - 1) / 1.7183;

        int icP = val;
        float ifP = val - icP;
        icP = icP % 3;

        cv::Vec3b color;
        if (icP == 0) color = cv::Vec3b(0, 0, 255 * ifP);
        if (icP == 1) color = cv::Vec3b(0, 255 * ifP, 255);
        if (icP == 2) color = cv::Vec3b(255 * ifP, 255, 255);

        EImg.at<cv::Vec3b>(i) = color;
        EImg16.at<ushort>(i) = 255 * 255 * (E[i] - Emin) / (Emax - Emin);
    }

    printf("Irradiance %f - %f\n", Emin, Emax);
    cv::imshow("lnE", EImg);

    if (saveTo != "") {
        cv::imwrite(saveTo + ".png", EImg);
        cv::imwrite(saveTo + "16.png", EImg16);
    }
}

void plotG(std::vector<double> G, std::string saveTo = "") {
    cv::Mat GImg = cv::Mat(256, 256, CV_32FC1);
    GImg.setTo(0);

    double min = 1e10, max = -1e10;

    for (int i = 0; i < 256; i++) {
        if (G[i] < min) min = G[i];
        if (G[i] > max) max = G[i];
    }

    for (int i = 0; i < 256; i++) {
        double val = 256 * (G[i] - min) / (max - min);
        for (int k = 0; k < 256; k++) {
            if (val < k)
                GImg.at<float>(k, i) = k - val;
        }
    }

    printf("Inv. Response %f - %f\n", min, max);
    cv::imshow("G", GImg);
    if (saveTo != "") cv::imwrite(saveTo, GImg * 255);
}


int iterations = 10;

int main() {
    std::string files = "../s30_CRF/";
    std::ifstream inFile, exFile;
    std::map<std::string, std::vector<float> > data;

    lazycsv::parser parser{files + "times.csv"};
    std::vector<std::string> imageFileNames;
    std::vector<double> exposureTimes; // in ms
    imageFileNames.reserve(1200);
    exposureTimes.reserve(1200);
    for (const auto row: parser) {
        const auto [frame, exposure] = row.cells(0, 1); // indexes must be in ascending order
        imageFileNames.emplace_back(std::string(frame.trimed()));
        exposureTimes.emplace_back(std::stod(std::string(exposure.trimed())));
    }

    std::string intrinsicsFile = (files + "intrinsics.yml");
    inFile.open(intrinsicsFile.c_str());
    if (!inFile) {
        fprintf(stderr, "failed to open '%s' for reading\n",
                intrinsicsFile.c_str());
    }
    parseYaml(inFile, data);
    cv::Mat distortionCoefficients = (cv::Mat_<double>(8, 1)
            << data["D1"][0], data["D1"][1], data["D1"][2], data["D1"][3], data["D1"][4], data["D1"][5], data["D1"][6], data["D1"][7]);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3)
            << data["M1"][0], data["M1"][1], data["M1"][2], data["M1"][3], data["M1"][4], data["M1"][5], data["M1"][6], data["M1"][7], data["M1"][8], data["M1"][9]);

    std::vector<cv::Mat> images;
    images.reserve(1200);
    printf("Loading images\n");
    for (size_t i = 0; i < imageFileNames.size() / 10; ++i) {
        images.emplace_back(cv::imread(files + ("images/" + imageFileNames[i] + ".png")));
    }
    printf("Loaded %zu images\n", images.size());

    int w = images[0].cols, h = images[0].rows;

    std::vector<double> E(w * h);  // scene irradiance
    std::vector<double> En(w * h); // scene irradiance
    std::vector<double> G(256);    // inverse response function
    // set starting scene irradiance to mean of all images.
    uint32_t v = 0;
    for (int i = 0; i < images.size(); i++) {
        for (int k = 0; k < w * h; k++) {
            //if(dataVec[i][k]==255) continue;
            E[k] += images[i].data[k];
            En[k]++;
        }
    }
    for (int k = 0; k < w * h; k++) {
        E[k] = E[k] / En[k];
    }

    if (-1 == system("rm -rf photoCalibResult")) printf("could not delete old photoCalibResult folder!\n");
    if (-1 == system("mkdir photoCalibResult")) printf("could not create photoCalibResult folder!\n");

    std::ofstream logFile;
    logFile.open("photoCalibResult/log.txt", std::ios::trunc | std::ios::out);
    logFile.precision(15);


    printf("init RMSE = %f! \t", rmse(G, E, exposureTimes, images, w*h )[0]);
    plotE(E,w,h, "photoCalibResult/E-0");
    cv::waitKey(100);

    for (int it = 0; it < iterations; it++) {
        printf("Iteration: %d/%d", it + 1, iterations);
        // optimize log inverse response function.
        double *GSum = new double[256];
        double *GNum = new double[256];
        memset(GSum, 0, 256 * sizeof(double));
        memset(GNum, 0, 256 * sizeof(double));
        for (int i = 0; i < images.size(); i++) {
            for (int k = 0; k < w * h; k++) {
                int b = images[i].data[k];
                if (b == 255) continue;
                GNum[b]++;
                GSum[b] += E[k] * exposureTimes[i];
            }
        }
        for (int i = 0; i < 256; i++) {
            G[i] = GSum[i] / GNum[i];
            if (!std::isfinite(G[i]) && i > 1)
                G[i] = G[i - 1] + (G[i - 1] - G[i - 2]);
        }
        delete[] GSum;
        delete[] GNum;
        printf("optG RMSE = %f! \t", rmse(G, E, exposureTimes, images, w * h)[0]);

        char buf[1000];
        snprintf(buf, 1000, "photoCalibResult/G-%d.png", it + 1);
        plotG(G, buf);


        // optimize scene irradiance function.
        double *ESum = new double[w * h];
        double *ENum = new double[w * h];
        memset(ESum, 0, w * h * sizeof(double));
        memset(ENum, 0, w * h * sizeof(double));
        for (int i = 0; i < images.size(); i++) {
            for (int k = 0; k < w * h; k++) {
                int b = images[i].data[k];
                if (b == 255) continue;
                ENum[k] += exposureTimes[i] * exposureTimes[i];
                ESum[k] += (G[b]) * exposureTimes[i];
            }
        }
        for (int i = 0; i < w * h; i++) {
            E[i] = ESum[i] / ENum[i];
            if (E[i] < 0) E[i] = 0;
        }

        delete[] ENum;
        delete[] ESum;
        printf("OptE RMSE = %f!  \t", rmse(G, E, exposureTimes, images, w * h)[0]);

        snprintf(buf, 1000, "photoCalibResult/E-%d", it + 1);
        plotE(E, w, h, buf);


        // rescale such that maximum response is 255 (fairly arbitrary choice).
        double rescaleFactor = 255.0 / G[255];
        for (int i = 0; i < w * h; i++) {
            E[i] *= rescaleFactor;
            if (i < 256) G[i] *= rescaleFactor;
        }
        Eigen::Vector2d err = rmse(G, E, exposureTimes, images, w * h);
        printf("resc RMSE = %f!  \trescale with %f!\n", err[0], rescaleFactor);

        //logFile << it << " " << n << " " << err[1] << " " << err[0] << "\n";

        cv::waitKey(100);
        printf("\n");
    }


    std::ofstream lg;
    lg.open("photoCalibResult/pcalib.txt", std::ios::trunc | std::ios::out);
    lg.precision(15);
    for (int i = 0; i < 256; i++)
        lg << G[i] << " ";
    lg << "\n";

    lg.flush();
    lg.close();


    return 0;
}
