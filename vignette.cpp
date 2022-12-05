//
// Created by magnus on 11/22/22.
//


#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"

#include <opencv2/aruco.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "Eigen/Core"
#include "Eigen/LU"
#include "lazycsv.h"
#include "CalibrationYaml.hh"
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <filesystem>
#include "calibration.h"

// reads interpolated element from a uchar* array
// SSE2 optimization possible
void distortCoordinates(float *in_x, float *in_y, int n) {
    /*
    float dist = inputCalibration[4];
    float d2t = 2.0f * tan(dist / 2.0f);

    // current camera parameters
    float fx = inputCalibration[0] * in_width;
    float fy = inputCalibration[1] * in_height;
    float cx = inputCalibration[2] * in_width - 0.5;
    float cy = inputCalibration[3] * in_height - 0.5;

    float ofx = outputCalibration[0]*out_width;
    float ofy = outputCalibration[1]*out_height;
    float ocx = outputCalibration[2]*out_width-0.5f;
    float ocy = outputCalibration[3]*out_height-0.5f;

    for(int i=0;i<n;i++)
    {
        float x = in_x[i];
        float y = in_y[i];
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;

        float r = sqrtf(ix*ix + iy*iy);
        float fac = (r==0 || dist==0) ? 1 : atanf(r * d2t)/(dist*r);

        ix = fx*fac*ix+cx;
        iy = fy*fac*iy+cy;

        in_x[i] = ix;
        in_y[i] = iy;
    }
     */
}


struct Image {
    float *data;
    int w = 0, h = 0;
    float exposure = 15439.0f / 1000.0f;

    Image(int w_, int h_) {
        w = w_;
        h = h_;
        data = new float[w_ * h_];
    }

    ~Image() {
        delete[] data;
    }
};

EIGEN_ALWAYS_INLINE float getInterpolatedElement(const cv::Mat &mat, const float x, const float y, const int width) {
    //stats.num_pixelInterpolations++;

    int ix = (int) x;
    int iy = (int) y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx * dy;
    const float *bp = reinterpret_cast<float *>(mat.data) + ix + iy * width;


    float res = dxdy * bp[1 + width]
                + (dy - dxdy) * bp[width]
                + (dx - dxdy) * bp[1]
                + (1 - dx - dy + dxdy) * bp[0];

    return res;
}

EIGEN_ALWAYS_INLINE float getInterpolatedElement(const float *mat, const float x, const float y, const int width) {
    //stats.num_pixelInterpolations++;

    int ix = (int) x;
    int iy = (int) y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx * dy;
    const float *bp = mat + ix + iy * width;


    float res = dxdy * bp[1 + width]
                + (dy - dxdy) * bp[width]
                + (dx - dxdy) * bp[1]
                + (1 - dx - dy + dxdy) * bp[0];

    return res;
}

void displayImage(float *I, int w, int h, std::string name) {
    float vmin = 1e10;
    float vmax = -1e10;

    for (int i = 0; i < w * h; i++) {
        if (vmin > I[i]) vmin = I[i];
        if (vmax < I[i]) vmax = I[i];
    }

    cv::Mat img = cv::Mat(h, w, CV_8UC3);

    for (int i = 0; i < w * h; i++) {
        if (isnanf(I[i])) img.at<cv::Vec3b>(i) = cv::Vec3b(0, 0, 255);
        else
            img.at<cv::Vec3b>(i) =
                    cv::Vec3b(255 * (I[i] - vmin) / (vmax - vmin),
                              255 * (I[i] - vmin) / (vmax - vmin),
                              255 * (I[i] - vmin) / (vmax - vmin));
    }

    printf("plane image values %f - %f!\n", vmin, vmax);
    cv::imshow(name, img);
    cv::imwrite("vignetteCalibResult/plane.png", img);
}

void displayImageV(float *I, int w, int h, std::string name) {
    cv::Mat img = cv::Mat(h, w, CV_8UC3);
    for (int i = 0; i < w * h; i++) {
        if (isnanf(I[i]))
            img.at<cv::Vec3b>(i) = cv::Vec3b(0, 0, 255);
        else {
            float c = 254 * I[i];
            img.at<cv::Vec3b>(i) = cv::Vec3b(c, c, c);
        }

    }
    cv::imshow(name, img);
}


// width of grid relative to marker (fac times marker size)
float facw = 3;
float fach = 3;
int maxIterations = 20;
int outlierTh = 15;
int maxAbsGrad = 255;


int main() {
    uint32_t gw = 1000, gh = 1000;

    // affine map from plane cordinates to grid coordinates.
    Eigen::Matrix3f K_p2idx = Eigen::Matrix3f::Identity();
    K_p2idx(0, 0) = gw / facw;
    K_p2idx(1, 1) = gh / fach;
    K_p2idx(0, 2) = gw / 2;
    K_p2idx(1, 2) = gh / 2;
    Eigen::Matrix3f K_p2idx_inverse = K_p2idx.inverse();

    std::cout << K_p2idx_inverse << std::endl;

    std::string files = "../s30_vignette/";
    std::ifstream inFile, exFile;
    std::map<std::string, std::vector<float> > data;

    float exposureTime = 15439.0f / 1000.0f;

    //lazycsv::parser parser{files + "times.csv"};
    std::vector<std::string> imageFileNames;
    std::vector<double> exposureTimes; // in ms
    imageFileNames.reserve(1200);
    exposureTimes.reserve(1200);
    /*
for (const auto row: parser) {
    const auto [frame, exposure] = row.cells(0, 1); // indexes must be in ascending order
    imageFileNames.emplace_back(std::string(frame.trimed()));
    exposureTimes.emplace_back(std::stod(std::string(exposure.trimed())) / 1000);
}
 */

    for (const auto &entry: std::filesystem::directory_iterator("../s30_vignette/constant_exposure/images"))
        imageFileNames.emplace_back(entry.path().filename());

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
    for (size_t i = 0; i < imageFileNames.size() / 4; ++i) {
        images.emplace_back(
                cv::imread(files + ("constant_exposure/images/" + imageFileNames[i]), cv::IMREAD_GRAYSCALE));
        if (images[i].data == NULL)
            throw std::runtime_error("Failed to load image: " + imageFileNames[i]);
    }
    printf("Loaded %zu images\n", images.size());
    mkdir("vignetteCalibResult", 0777);

    // load the inverse response function
    std::string pCalibFile = "../result/photoCalibResult/pcalib.txt";
    std::ifstream f(pCalibFile.c_str());
    std::string line;
    std::getline(f, line);
    std::istringstream l1i(line);
    std::vector<float> GInvvec = std::vector<float>(std::istream_iterator<float>(l1i), std::istream_iterator<float>());
    float GInv[256];

    if (GInvvec.size() != 256) {
        printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected 256!\n",
               (int) GInvvec.size());
    }

    for (int i = 0; i < 256; i++)
        GInv[i] = GInvvec[i];

    float min=GInv[0];
    float max=GInv[255];
    for(float & i : GInv) i = 255.0f * (i - min) / (max-min);			// make it to 0..255 => 0..255.

    bool isGood = true;
    for (int i = 0; i < 255; i++) {
        if (GInv[i + 1] <= GInv[i]) {
            printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
            isGood = false;
        }
    }
    if (isGood)
        printf("Loaded camera inverse response function\n");

    int w = images[0].cols, h = images[0].rows;

    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    std::vector<float *> p2imgX;
    std::vector<float *> p2imgY;

    float meanExposure = exposureTime;
    for (int i = 0; i < images.size(); ++i)
        meanExposure += exposureTimes[i];

    if (meanExposure == 0)
        meanExposure = 1;

    std::vector<float *> imageList;

    for (int i = 0; i < images.size(); i++) {
        cv::Mat inputImage = images[i].clone();
        std::vector<int> markerIds;
        cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        cv::imshow("DetectorImage", inputImage);

        if (markerIds.size() != 1) {
            printf("Did not find marker in image %d\n", i);
            continue;
        }

        std::vector<cv::Point2f> ptsP;
        std::vector<cv::Point2f> ptsI;
        ptsI.emplace_back(markerCorners[0][0].x, markerCorners[0][0].y);
        ptsI.emplace_back(cv::Point2f(markerCorners[0][1].x, markerCorners[0][1].y));
        ptsI.emplace_back(cv::Point2f(markerCorners[0][2].x, markerCorners[0][2].y));
        ptsI.emplace_back(cv::Point2f(markerCorners[0][3].x, markerCorners[0][3].y));
        // Clockwise corners from top left
        ptsP.emplace_back(cv::Point2f(-0.5, 0.5));
        ptsP.emplace_back(cv::Point2f(0.5, 0.5));
        ptsP.emplace_back(0.5, -0.5);
        ptsP.emplace_back(cv::Point2f(-0.5, -0.5));

        cv::Mat Hcv = cv::findHomography(ptsP, ptsI);
        Eigen::Matrix3f H;
        H(0, 0) = Hcv.at<double>(0, 0);
        H(0, 1) = Hcv.at<double>(0, 1);
        H(0, 2) = Hcv.at<double>(0, 2);
        H(1, 0) = Hcv.at<double>(1, 0);
        H(1, 1) = Hcv.at<double>(1, 1);
        H(1, 2) = Hcv.at<double>(1, 2);
        H(2, 0) = Hcv.at<double>(2, 0);
        H(2, 1) = Hcv.at<double>(2, 1);
        H(2, 2) = Hcv.at<double>(2, 2);

        // Output image
        //cv::Mat im_out;
        // Warp source image to destination based on homography
        //cv::warpPerspective(inputImage, im_out, Hcv, inputImage.size());
        //cv::imshow("Warped", im_out);

        Eigen::Matrix3f HK = H * K_p2idx_inverse;

        auto *plane2imgX = new float[gw * gh];
        auto *plane2imgY = new float[gw * gh];
        // every point in the plante gets transformed using the homography
        int idx = 0;
        for (int y = 0; y < gh; y++)
            for (int x = 0; x < gw; x++) {
                Eigen::Vector3f pp = HK * Eigen::Vector3f(x, y, 1);
                plane2imgX[idx] = pp[0] / pp[2];
                plane2imgY[idx] = pp[1] / pp[2];
                idx++;
            }
        auto imgRaw = Image(w, h);
        int v = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                imgRaw.data[v] = (float) inputImage.at<uchar>(y, x);
                v++;
            }
        }

        for (int x = 0; x < gw * gh; x++)
            distortCoordinates(&plane2imgX[x], &plane2imgY[x]);

        auto *image = new float[w * h];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                image[x + y * w] = imgRaw.data[x + y * w];

                // Apply the inverse camera response function
                image[x + y * w] = GInv[static_cast<unsigned char>(image[x + y * w])];
            }
        }

        for (int y = 2; y < h - 2; y++)
            for (int x = 2; x < w - 2; x++) {
                for (int deltax = -2; deltax < 3; deltax++)
                    for (int deltay = -2; deltay < 3; deltay++) {
                        if (fabsf(image[x + y * w] - image[x + deltax + (y + deltay) * w]) > maxAbsGrad) {
                            image[x + y * w] = NAN;
                            image[x + deltax + (y + deltay) * w] = NAN;
                        }
                    }
            }

        imageList.push_back(image);
        cv::Mat dbgImg(imgRaw.h, imgRaw.w, CV_8UC3);
        for (int pix = 0; pix < imgRaw.w * imgRaw.h; pix++) {
            dbgImg.at<cv::Vec3b>(pix) = cv::Vec3b(imgRaw.data[pix], imgRaw.data[pix], imgRaw.data[pix]);
        }

        for (int x = 0; x <= gw; x += 200)
            for (int y = 0; y <= gh; y += 10) {
                int idxS = (x < gw ? x : gw - 1) + (y < gh ? y : gh - 1) * gw;
                int idxT = (x < gw ? x : gw - 1) + ((y + 10) < gh ? (y + 10) : gh - 1) * gw;

                int u_dS = plane2imgX[idxS] + 0.5;
                int v_dS = plane2imgY[idxS] + 0.5;

                int u_dT = plane2imgX[idxT] + 0.5;
                int v_dT = plane2imgY[idxT] + 0.5;

                if (u_dS >= 0 && v_dS >= 0 && u_dS < w && v_dS < h && u_dT >= 0 && v_dT >= 0 && u_dT < w && v_dT < h)
                    cv::line(dbgImg, cv::Point(u_dS, v_dS), cv::Point(u_dT, v_dT), cv::Scalar(0, 0, 255), 10,
                             cv::LINE_AA);
            }


        for (int x = 0; x <= gw; x += 10)
            for (int y = 0; y <= gh; y += 200) {
                int idxS = (x < gw ? x : gw - 1) + (y < gh ? y : gh - 1) * gw;
                int idxT = ((x + 10) < gw ? (x + 10) : gw - 1) + (y < gh ? y : gh - 1) * gw;

                int u_dS = plane2imgX[idxS] + 0.5;
                int v_dS = plane2imgY[idxS] + 0.5;

                int u_dT = plane2imgX[idxT] + 0.5;
                int v_dT = plane2imgY[idxT] + 0.5;

                if (u_dS >= 0 && v_dS >= 0 && u_dS < w && v_dS < h && u_dT >= 0 && v_dT >= 0 && u_dT < w && v_dT < h)
                    cv::line(dbgImg, cv::Point(u_dS, v_dS), cv::Point(u_dT, v_dT), cv::Scalar(0, 0, 255), 10,
                             cv::LINE_AA);
            }


        for (int x = 0; x < gw; x++)
            for (int y = 0; y < gh; y++) {
                int u_d = plane2imgX[x + y * gw] + 0.5;
                int v_d = plane2imgY[x + y * gw] + 0.5;

                if (!(u_d > 1 && v_d > 1 && u_d < w - 2 && v_d < h - 2)) {
                    plane2imgX[x + y * gw] = NAN;
                    plane2imgY[x + y * gw] = NAN;
                }
            }

        cv::imshow("inRaw", dbgImg);

        if (rand() % 40 == 0) {
            char buf[1000];
            snprintf(buf, 1000, "vignetteCalibResult/img%d.png", i);
            cv::imwrite(buf, dbgImg);
        }
        p2imgX.push_back(plane2imgX);
        p2imgY.push_back(plane2imgY);

        //delete[] plane2imgX;
        //delete[] plane2imgY;
        //delete[] image;

        std::cout << "Iterated image: " << i << std::endl;
        cv::waitKey(1);
    }

    std::ofstream logFile;
    logFile.open("vignetteCalibResult/log.txt", std::ios::trunc | std::ios::out);
    logFile.precision(15);

    float *planeColor = new float[gw * gh];
    float *planeColorFF = new float[gw * gh];
    float *planeColorFC = new float[gw * gh];
    float *vignetteFactor = new float[h * w];
    float *vignetteFactorTT = new float[h * w];
    float *vignetteFactorCT = new float[h * w];

    // initialize vignette factors to 1.
    for (int i = 0; i < h * w; i++) vignetteFactor[i] = 1;
    double E = 0;
    double R = 0;
    for (int it = 0; it < maxIterations; it++) {
        int oth2 = outlierTh * outlierTh;
        if (it < maxIterations / 2)
            oth2 = 10000 * 10000;

        // ============================ optimize planeColor ================================
        memset(planeColorFF, 0, gw * gh * sizeof(float));
        memset(planeColorFC, 0, gw * gh * sizeof(float));
        E = 0;
        R = 0;

        // for each plane pixel, it's optimum is at sum(CF)/sum(FF)
        for (int img = 0; img < imageList.size(); img++)    // for all images
        {
            float *plane2imgX = p2imgX[img];
            float *plane2imgY = p2imgY[img];


            for (int pi = 0; pi < gw * gh; pi++)        // for all plane points
            {
                if (isnanf(plane2imgX[pi])) continue;

                // get vignetted color at that point, and add to build average.
                float color = getInterpolatedElement(imageList[img], plane2imgX[pi], plane2imgY[pi], w);
                float fac = getInterpolatedElement(vignetteFactor, plane2imgX[pi], plane2imgY[pi], w);

                if (isnanf(fac)) continue;
                if (isnanf(color)) continue;

                double residual = (double) ((color - planeColor[pi] * fac) * (color - planeColor[pi] * fac));
                if (abs(residual) > oth2) {
                    E += oth2;
                    R++;
                    continue;
                }

                planeColorFF[pi] += fac * fac;
                planeColorFC[pi] += color * fac;

                if (isnanf(planeColor[pi])) continue;
                E += residual;
                R++;
            }
        }

        for (int pi = 0; pi < gw * gh; pi++)        // for all plane points
        {
            if (planeColorFF[pi] < 1)
                planeColor[pi] = NAN;
            else
                planeColor[pi] = planeColorFC[pi] / planeColorFF[pi];
        }
        displayImage(planeColor, gw, gh, "Plane");

        printf("%f residual terms => %f\n", R, sqrtf(E / R));

        // ================================ optimize vignette =======================================
        memset(vignetteFactorTT, 0, w * h * sizeof(float));
        memset(vignetteFactorCT, 0, w * h * sizeof(float));
        E = 0;
        R = 0;

        for (int img = 0; img < imageList.size(); img++)    // for all images
        {
            float *plane2imgX = p2imgX[img];
            float *plane2imgY = p2imgY[img];

            for (int pi = 0; pi < gw * gh; pi++)        // for all plane points
            {
                if (isnanf(plane2imgX[pi])) continue;
                float x = plane2imgX[pi];
                float y = plane2imgY[pi];

                float colorImage = getInterpolatedElement(imageList[img], x, y, w);
                float fac = getInterpolatedElement(vignetteFactor, x, y, w);
                float colorPlane = planeColor[pi];

                if (isnanf(colorPlane)) continue;
                if (isnanf(colorImage)) continue;

                double residual = (double) ((colorImage - colorPlane * fac) * (colorImage - colorPlane * fac));
                if (abs(residual) > oth2) {
                    E += oth2;
                    R++;
                    continue;
                }


                int ix = (int) x;
                int iy = (int) y;
                float dx = x - ix;
                float dy = y - iy;
                float dxdy = dx * dy;

                vignetteFactorTT[ix + iy * w + 0] += (1 - dx - dy + dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * w + 1] += (dx - dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * w + w] += (dy - dxdy) * colorPlane * colorPlane;
                vignetteFactorTT[ix + iy * w + 1 + w] += dxdy * colorPlane * colorPlane;

                vignetteFactorCT[ix + iy * w + 0] += (1 - dx - dy + dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * w + 1] += (dx - dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * w + w] += (dy - dxdy) * colorImage * colorPlane;
                vignetteFactorCT[ix + iy * w + 1 + w] += dxdy * colorImage * colorPlane;

                if (isnanf(fac)) continue;
                E += residual;
                R++;
            }
        }

        float maxFac = 0;
        for (int pi = 0; pi < h * w; pi++)        // for all plane points
        {
            if (vignetteFactorTT[pi] < 1)
                vignetteFactor[pi] = NAN;
            else {
                vignetteFactor[pi] = vignetteFactorCT[pi] / vignetteFactorTT[pi];
                if (vignetteFactor[pi] > maxFac) maxFac = vignetteFactor[pi];
            }
        }

        printf("%f residual terms => %f\n", R, sqrtf(E / R));

        // normalize to vignette max. factor 1.
        for (int pi = 0; pi < h * w; pi++)
            vignetteFactor[pi] /= maxFac;

        logFile << it << " " << imageList.size() << " " << R << " " << sqrtf(E / R) << "\n";
        // dilate & smoothe vignette by 4 pixel for output.
        // does not change anything in the optimization; uses vignetteFactorTT and vignetteFactorCT for temporary storing
        {
            memcpy(vignetteFactorTT, vignetteFactor, sizeof(float) * h * w);
            for (int dilit = 0; dilit < 4; dilit++) {
                memcpy(vignetteFactorCT, vignetteFactorTT, sizeof(float) * h * w);
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++) {
                        int idx = x + y * w;
                        {
                            float sum = 0, num = 0;
                            if (x < w - 1 && y < h - 1 && !isnanf(vignetteFactorCT[idx + 1 + w])) {
                                sum += vignetteFactorCT[idx + 1 + w];
                                num++;
                            }
                            if (x < w - 1 && !isnanf(vignetteFactorCT[idx + 1])) {
                                sum += vignetteFactorCT[idx + 1];
                                num++;
                            }
                            if (x < w - 1 && y > 0 && !isnanf(vignetteFactorCT[idx + 1 - w])) {
                                sum += vignetteFactorCT[idx + 1 - w];
                                num++;
                            }

                            if (y < h - 1 && !isnanf(vignetteFactorCT[idx + w])) {
                                sum += vignetteFactorCT[idx + w];
                                num++;
                            }
                            if (!isnanf(vignetteFactorCT[idx])) {
                                sum += vignetteFactorCT[idx];
                                num++;
                            }
                            if (y > 0 && !isnanf(vignetteFactorCT[idx - w])) {
                                sum += vignetteFactorCT[idx - w];
                                num++;
                            }

                            if (y < h - 1 && x > 0 && !isnanf(vignetteFactorCT[idx - 1 + w])) {
                                sum += vignetteFactorCT[idx - 1 + w];
                                num++;
                            }
                            if (x > 0 && !isnanf(vignetteFactorCT[idx - 1])) {
                                sum += vignetteFactorCT[idx - 1];
                                num++;
                            }
                            if (y > 0 && x > 0 && !isnanf(vignetteFactorCT[idx - 1 - w])) {
                                sum += vignetteFactorCT[idx - 1 - w];
                                num++;
                            }

                            if (num > 0) vignetteFactorTT[idx] = sum / num;
                        }
                    }
            }
            {
                displayImageV(vignetteFactorTT, w, h, "VignetteSmoothed");
                cv::Mat wrap = cv::Mat(h, w, CV_32F, vignetteFactorTT) * 254.9 * 254.9;
                cv::Mat wrap16;
                wrap.convertTo(wrap16, CV_16U, 1, 0);
                cv::imwrite("vignetteCalibResult/vignetteSmoothed.png", wrap16);
                cv::waitKey(50);
            }
            {
                displayImageV(vignetteFactor, w, h, "VignetteOrg");
                cv::Mat wrap = cv::Mat(h, w, CV_32F, vignetteFactor) * 254.9 * 254.9;
                cv::Mat wrap16;
                wrap.convertTo(wrap16, CV_16U, 1, 0);
                cv::imwrite("vignetteCalibResult/vignette.png", wrap16);
                cv::waitKey(50);
            }
        }
    }


    logFile.flush();
    logFile.close();

    return 0;
}