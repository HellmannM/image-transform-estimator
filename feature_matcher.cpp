#include <array>
#include <iostream>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp> // cvt::cuda::cvtColor
#include <opencv2/imgproc.hpp> // cv::cvtColor
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/features2d.hpp> // cv::SIFT, cv::ORB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp> // cv::xfeatures2d::SURF
#if VSNRAY_COMMON_HAVE_CUDA
#include <opencv2/cudafeatures2d.hpp> // cv::cuda::ORB
#endif

#include "feature_matcher.h"


template <typename Detector, typename Descriptor, typename Matcher>
feature_matcher<Detector, Descriptor, Matcher>::feature_matcher()
    : detector(Detector::create())
    , descriptor(Descriptor::create())
    , matcher(cv::BFMatcher::create(cv::NORM_L2, true))
    , matcher_initialized(false)
    , reference_descriptors()
    , reference_keypoints()
    {};

template<>
feature_matcher<cv::xfeatures2d::SURF, cv::SIFT, cv::BFMatcher>::feature_matcher()
    : detector(cv::xfeatures2d::SURF::create())
    , descriptor(cv::SIFT::create())
    , matcher(cv::BFMatcher::create(cv::NORM_L2, true))
    , matcher_initialized(false)
    , reference_descriptors()
    , reference_keypoints()
    {};

template<>
feature_matcher<cv::ORB, cv::ORB, cv::BFMatcher>::feature_matcher()
    : detector(cv::ORB::create(                             // default values
            /*int nfeatures     */ 5000,                    // 500
            /*float scaleFactor */ 1.1f,                    // 1.2f
            /*int nlevels       */ 15,                      // 8
            /*int edgeThreshold */ 10,                      // 31
            /*int firstLevel    */ 0,                       // 0
            /*int WTA_K         */ 2,                       // 2
            /*int scoreType     */ cv::ORB::HARRIS_SCORE,   // cv::ORB::HARRIS_SCORE
            /*int patchSize     */ 31,                      // 31
            /*int fastThreshold */ 10                       // 20
      ))
    , descriptor(cv::ORB::create())
    , matcher(cv::BFMatcher::create(cv::NORM_HAMMING, true))
    , matcher_initialized(false)
    , reference_descriptors()
    , reference_keypoints()
    {};

template <typename Detector, typename Descriptor, typename Matcher>
void feature_matcher<Detector, Descriptor, Matcher>::set_reference_image(
    const std::vector<uint8_t>& data, size_t width, size_t height, PIXEL_TYPE pixel_type)
{
    if (pixel_type != PIXEL_TYPE::RGBA)
    {
        std::cerr << "Feature matcher: Unsupported pixel type.\n";
        return;
    }

    #define SHUFFLE
    #ifdef SHUFFLE
    size_t bpp = 4;
    std::vector<uint8_t> shuffled;
    shuffled.resize(width * height * bpp);
    for (size_t y=0; y<height; ++y)
    {
        for (size_t x=0; x<width; ++x)
        {
            shuffled[4 * ((height - y - 1) * width + x)    ] = data.data()[4*(y * width + x)    ];
            shuffled[4 * ((height - y - 1) * width + x) + 1] = data.data()[4*(y * width + x) + 1];
            shuffled[4 * ((height - y - 1) * width + x) + 2] = data.data()[4*(y * width + x) + 2];
            shuffled[4 * ((height - y - 1) * width + x) + 3] = data.data()[4*(y * width + x) + 3];
        }
    }
    const auto pixels = reinterpret_cast<void*>(shuffled.data());
    #else
    const auto pixels = reinterpret_cast<void*>(data.data());
    #endif

    const auto reference_image = cv::Mat(height, width, CV_8UC4, pixels);
    init(reference_image);
}
    
template <typename Detector, typename Descriptor, typename Matcher>
void feature_matcher<Detector, Descriptor, Matcher>::init(const cv::Mat& reference_image)
{
    reference_keypoints.clear();
    detector->detect(reference_image, reference_keypoints, cv::noArray());
    descriptor->compute(reference_image, reference_keypoints, reference_descriptors);
    matcher->clear();
    matcher->add(reference_descriptors);
    matcher_initialized = true;
    std::cout << "Found " << reference_descriptors.size() << " descriptors.\n";
}
    
template <typename Detector, typename Descriptor, typename Matcher>
void feature_matcher<Detector, Descriptor, Matcher>::match(
    const std::vector<uint8_t>& data, size_t width, size_t height, PIXEL_TYPE pixel_type)
{
    if (pixel_type != PIXEL_TYPE::RGBA)
    {
        std::cerr << "Feature matcher: Unsupported pixel type.\n";
        return;
    }
    //#define SHUFFLE2
    #ifdef SHUFFLE2
    size_t bpp = 4;
    std::vector<uint8_t> shuffled;
    shuffled.resize(width * height * bpp);
    for (size_t y=0; y<height; ++y)
    {
        for (size_t x=0; x<width; ++x)
        {
            shuffled[4 * ((height - y - 1) * width + x)    ] = data.data()[4*(y * width + x)    ];
            shuffled[4 * ((height - y - 1) * width + x) + 1] = data.data()[4*(y * width + x) + 1];
            shuffled[4 * ((height - y - 1) * width + x) + 2] = data.data()[4*(y * width + x) + 2];
            shuffled[4 * ((height - y - 1) * width + x) + 3] = data.data()[4*(y * width + x) + 3];
        }
    }
    const auto pixels = reinterpret_cast<void*>(shuffled.data());
    #else
    const auto pixels = reinterpret_cast<void*>(data.data());
    #endif
    const auto current_image = cv::Mat(height, width, CV_8UC4, pixels);

    std::vector<cv::KeyPoint> current_keypoints;
    match_result = match_result_t();

    if (!matcher_initialized) return;
    cv::Mat current_descriptors;
    detector->detect(current_image, current_keypoints, cv::noArray());
    descriptor->compute(current_image, current_keypoints, current_descriptors);
    matcher->match(current_descriptors, match_result.matches, cv::noArray());
    match_result.num_ref_descriptors = reference_descriptors.size().height;
    match_result.reference_keypoints = reference_keypoints;
    match_result.query_keypoints = current_keypoints;

    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    //cv::Mat img;
    ////cv::drawMatches(current_image, current_keypoints, reference_image, reference_keypoints, result.matches, img);
    //cv::drawMatches(current_image, current_keypoints, current_image, reference_keypoints, result.matches, img);
    //cv::imshow("Display Image", img);
    //cv::waitKey(0);
}

#if VSNRAY_COMMON_HAVE_CUDA
template<>
feature_matcher<cv::cuda::ORB, cv::cuda::ORB, cv::cuda::DescriptorMatcher>::feature_matcher()
    : detector(cv::cuda::ORB::create(                             // default values
            /*int nfeatures     */ 5000,                    // 500
            /*float scaleFactor */ 1.1f,                    // 1.2f
            /*int nlevels       */ 15,                      // 8
            /*int edgeThreshold */ 10,                      // 31
            /*int firstLevel    */ 0,                       // 0
            /*int WTA_K         */ 2,                       // 2
            /*int scoreType     */ cv::ORB::HARRIS_SCORE,   // cv::ORB::HARRIS_SCORE
            /*int patchSize     */ 31,                      // 31
            /*int fastThreshold */ 10                       // 20
      ))
    , descriptor(cv::cuda::ORB::create())
    , matcher(cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING))
    , matcher_initialized(false)
    , reference_descriptors()
    , reference_keypoints()
    {};

template<>
void feature_matcher<cv::cuda::ORB, cv::cuda::ORB, cv::cuda::DescriptorMatcher>::init(const cv::Mat& reference_image)
{
    reference_keypoints.clear();
    cv::cuda::GpuMat gpu_reference_image_color(reference_image);
    cv::cuda::GpuMat gpu_reference_image;
    cv::cuda::cvtColor(gpu_reference_image_color, gpu_reference_image, cv::COLOR_RGBA2GRAY);
    detector->detectAndCompute(gpu_reference_image, cv::noArray(), reference_keypoints, gpu_reference_descriptors);
    matcher->clear();
    matcher->add({gpu_reference_descriptors});
    matcher_initialized = true;
    std::cout << "Found " << gpu_reference_descriptors.size() << " descriptors.\n";
}

//template<>
//void feature_matcher<cv::cuda::ORB, cv::cuda::ORB, cv::cuda::DescriptorMatcher>::match(const cv::Mat& current_image)
//{
//    match_result = match_result_t();
//    std::vector<cv::KeyPoint> current_keypoints;
//    cv::cuda::GpuMat gpu_current_descriptors;
//    cv::cuda::GpuMat gpu_current_image_color(current_image);
//    cv::cuda::GpuMat gpu_current_image;
//    cv::cuda::cvtColor(gpu_current_image_color, gpu_current_image, cv::COLOR_RGBA2GRAY);
//
//    detector->detectAndCompute(gpu_current_image, cv::noArray(), current_keypoints, gpu_current_descriptors);
//    if (!matcher_initialized) return;
//    matcher->match(gpu_current_descriptors, match_result.matches);
//
//    match_result.num_ref_descriptors = gpu_reference_descriptors.size().height;
//    match_result.reference_keypoints = reference_keypoints;
//    match_result.query_keypoints = current_keypoints;
//}
#endif
    
template <typename Detector, typename Descriptor, typename Matcher>
void feature_matcher<Detector, Descriptor, Matcher>::calibrate(size_t width, size_t height, float fovy, float aspect)
{
    double fx = 0.5 * ((double)width - 1) / std::tan(0.5 * fovy * aspect); // fx=444.661
    double fy = 0.5 * ((double)height - 1) / std::tan(0.5 * fovy); // fy=462.322
    //TODO fy seems to be slightly off...
    fy *= 1.002612;
    double cx = ((double)width - 1) / 2.0; // (500-1)/2=249.5
    double cy = ((double)height - 1) / 2.0; // (384-1)/2=191.5
    fx = fy; //TODO doesn't like fx, works good with fy?!
    double camera_matrix_data[] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    // opencv stores in row-major order
    camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_data);
}

template <typename Detector, typename Descriptor, typename Matcher>
bool feature_matcher<Detector, Descriptor, Matcher>::update_camera(
        const std::vector<float>& depth,
        const std::vector<float>& depth3D,
        size_t width,
        size_t height,
        DEPTH_TYPE depth_type,
        DEPTH_TYPE depth3D_type,
        std::array<float, 3>& eye,
        std::array<float, 3>& center,
        std::array<float, 3>& up)
{
    if (depth3D_type != DEPTH_TYPE::FLOAT3)
    {
        std::cerr << "Feature matcher: Unsupported depth type.\n";
        return false;
    }
    //#define SHUFFLE3
    #ifdef SHUFFLE3
    std::vector<float> shuffled;
    shuffled.resize(width * height * 3);
    for (size_t y=0; y<height; ++y)
        for (size_t x=0; x<width; ++x)
        {
            shuffled[3 * ((height - y - 1) * width + x)    ] = depth3D[3*(y * width + x)    ];
            shuffled[3 * ((height - y - 1) * width + x) + 1] = depth3D[3*(y * width + x) + 1];
            shuffled[3 * ((height - y - 1) * width + x) + 2] = depth3D[3*(y * width + x) + 2];
        }
    const auto& depth3D = shuffled;
    #else
    const auto& depth3D_ = depth3D;
    #endif

    // calc dist eye to center for later use
    auto distance = cv::norm(cv::Point3f(eye[0], eye[1], eye[2]) - cv::Point3f(center[0], center[1], center[2]));

    auto good_matches = match_result.good_matches(50.f);
    std::cout << good_matches.size() << " good matches\n";
    constexpr size_t min_good_matches {6};
    if (good_matches.size() < min_good_matches)
    {
        std::cerr << "ERROR: found only " << good_matches.size()
                  << " good matches (minimum is " << min_good_matches
                  << "). Aborting search...\n";
        return false;
    }
    std::sort(good_matches.begin(), good_matches.end(),
              [](const cv::DMatch& lhs, const cv::DMatch& rhs){return lhs.distance < rhs.distance;});
    std::vector<cv::Point2f> reference_points;
    std::vector<cv::Point2f> query_points;
    constexpr size_t num_points_for_solvepnp {min_good_matches};
    //TODO took only num_points_for_solvepnp best points. Do filtering after depth estimation?
    //taking all for now
    //for (size_t i=0; i<std::min(num_points_for_solvepnp, good_matches.size()); ++i)
    for (size_t i=0; i<good_matches.size(); ++i)
    {
        reference_points.push_back(match_result.reference_keypoints[good_matches[i].trainIdx].pt);
        query_points.push_back(match_result.query_keypoints[good_matches[i].queryIdx].pt);
    }

    // get reprojection
    std::vector<cv::Point3f> query_coords;
    std::vector<cv::Point2f> reference_points_filtered;
    for (size_t i=0; i<query_points.size(); ++i)
    {
        auto& p = query_points[i];
        auto index = p.y * width + p.x;
        cv::Point3f coord {depth3D_[index], depth3D_[index+1], depth3D_[index+2]};
        if (depth[index] <= 0.f)
        { // depth estimation returned with low contribution value
            continue;
        }
//#define INV_Y
#define INV_Z
#ifdef INV_Y
        coord.y = -coord.y;
#endif
#ifdef INV_Z
        coord.z = -coord.z;
#endif
        query_coords.push_back(coord);
        reference_points_filtered.push_back(reference_points[i]);
    }

    if (query_coords.size() < num_points_for_solvepnp)
    {
        std::cerr << "ERROR: found only " << query_coords.size()
                  << " suitable coords. Aborting...\n";
        return false;
    }

    // solve
    cv::Mat rotation;
    cv::Mat translation = cv::Mat(3, 1, CV_64FC1, 0.0);
    translation.at<double>(0) = eye[0];
    translation.at<double>(1) = eye[1];
    translation.at<double>(2) = eye[2];
#if 1
    cv::solvePnP(
            query_coords,
            reference_points_filtered,
            camera_matrix,
            std::vector<double>(), // distCoeffs
            rotation,
            translation,
            false, // useExtrinsicGuess = false
    	    cv::SOLVEPNP_ITERATIVE
    	    //cv::SOLVEPNP_P3P
    	    //cv::SOLVEPNP_AP3P
    	    //cv::SOLVEPNP_SQPNP
    	    //cv::SOLVEPNP_EPNP
    );
#else
    cv::solvePnPRansac(
            query_coords,
            reference_points_filtered,
            camera_matrix,
            std::vector<double>(), // distCoeffs
            rotation,
            translation,
            false, // useExtrinsicGuess = false
            1000, // iterationsCount = 100
            0.8, // reprojectionError = 8.0
            0.6, // confidence = 0.99,
            cv::noArray(), // inliers = noArray(),
            cv::SOLVEPNP_ITERATIVE
            //cv::SOLVEPNP_IPPE // flags = SOLVEPNP_ITERATIVE
    );
#endif
    //std::cout << "rotation\n" << rotation << "\ntranslation\n" << translation << "\n";
    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation, rotation_matrix);

    // camera eye
    cv::Mat eye_cv = cv::Mat(3, 1, CV_32F, eye);
    cv::Mat new_eye = -1.0 * rotation_matrix.t() * translation;
#ifdef INV_Y
    new_eye.at<float>(1) = -new_eye.at<float>(1);
#endif
#ifdef INV_Z
    new_eye.at<float>(2) = -new_eye.at<float>(2);
#endif
    eye[0] = new_eye.at<float>(0);
    eye[1] = new_eye.at<float>(1);
    eye[2] = new_eye.at<float>(2);

    // camera up
    cv::Mat default_up = cv::Mat(3, 1, CV_32F);
    default_up.at<float>(0) = 0.f;
    default_up.at<float>(1) = 1.f;
    default_up.at<float>(2) = 0.f;
    cv::Mat new_up = rotation_matrix.t() * default_up;
    cv::Mat new_up_normalized;
    cv::normalize(new_up, new_up_normalized);
#ifdef INV_Y
    new_up_normalized.at<float>(1) = -new_up_normalized.at<float>(1);
#endif
#ifdef INV_Z
    new_up_normalized.at<float>(2) = -new_up_normalized.at<float>(2);
#endif
    up[0] = new_up_normalized.at<float>(0);
    up[1] = new_up_normalized.at<float>(1);
    up[2] = new_up_normalized.at<float>(2);

    // camera dir
    cv::Mat default_z = cv::Mat(3, 1, CV_32F);
    default_z.at<float>(0) = 0.f;
    default_z.at<float>(1) = 0.f;
    default_z.at<float>(2) = 1.f;
    cv::Mat new_dir = rotation_matrix.t() * default_z;
    cv::Mat new_dir_normalized;
    cv::normalize(new_dir, new_dir_normalized);
#ifdef INV_Y
    new_dir_normalized.at<float>(1) = -new_dir_normalized.at<float>(1);
#endif
#ifdef INV_Z
    new_dir_normalized.at<float>(2) = -new_dir_normalized.at<float>(2);
#endif
    
    // camera center
    cv::Point3f new_center = new_eye + distance * new_dir_normalized;
    center[0] = new_center.x;
    center[1] = new_center.y;
    center[2] = new_center.z;

    return true;
}