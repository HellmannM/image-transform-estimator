#include <array>
#include <cmath>
#include <exception>
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


//using PIXEL_TYPE = image_transform_estimator::PIXEL_TYPE;
//using IMAGE_TYPE = image_transform_estimator::IMAGE_TYPE;

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
    , good_match_threshold(80.f)
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
    , good_match_threshold(50.f)
    {};

template <typename Detector, typename Descriptor, typename Matcher>
std::vector<uint8_t> feature_matcher<Detector, Descriptor, Matcher>::swizzle_image(
    const void* data, size_t width, size_t height, PIXEL_TYPE pixel_type)
{
    size_t bpp{0};
    switch (pixel_type)
    {
    case PIXEL_TYPE::R8:
        bpp = 1;
        break;
    case PIXEL_TYPE::RGB8:
        bpp = 3;
        break;
    case PIXEL_TYPE::F32:
    case PIXEL_TYPE::RGBA8:
        bpp = 4;
        break;
    case PIXEL_TYPE::F32X3: // assuming packed float3
        bpp = 3 * 4;
        break;
    default:
        std::cerr << "Error in feature_matcher: Unsupported pixel type.\n";
        assert(false);
        return{};
    }
    std::vector<uint8_t> swizzled;
    swizzled.resize(width * height * bpp);
    const auto data_u8 = reinterpret_cast<const uint8_t*>(data);
    for (size_t y=0; y<height; ++y)
    {
        for (size_t x=0; x<width; ++x)
        {
            for (size_t b=0; b<bpp; ++b)
            {
                auto yy = height - y - 1;
                auto xx = width - x - 1;
                swizzled[bpp * (yy * width + xx) + b] = data_u8[bpp * (y * width + x) + b];
            }
        }
    }
    return std::move(swizzled);
}

template <typename Detector, typename Descriptor, typename Matcher>
void feature_matcher<Detector, Descriptor, Matcher>::set_image(
    const void* data,
    size_t width,
    size_t height,
    PIXEL_TYPE pixel_type,
    IMAGE_TYPE image_type,
    bool swizzle)
{
    const void* pixels{nullptr};
    std::vector<uint8_t> swizzled;
    if (swizzle)
    {
        swizzled = swizzle_image(data, width, height, pixel_type);
        pixels = reinterpret_cast<void*>(swizzled.data());
    } else
    {
        pixels = data;
    }

    std::vector<uint8_t> remapped;
    if (pixel_type == PIXEL_TYPE::R8)
    {
        remapped.resize(width * height * 4);
        for (size_t y=0; y<height; ++y)
        {
            for (size_t x=0; x<width; ++x)
            {
                size_t i = (y * width + x);
                remapped[4 * i    ] = reinterpret_cast<const uint8_t*>(pixels)[i];
                remapped[4 * i + 1] = reinterpret_cast<const uint8_t*>(pixels)[i];
                remapped[4 * i + 2] = reinterpret_cast<const uint8_t*>(pixels)[i];
                remapped[4 * i + 3] = 255;
            }
        }
        pixels = reinterpret_cast<void*>(remapped.data());
        pixel_type = PIXEL_TYPE::RGBA8;
    }
    else if (pixel_type == PIXEL_TYPE::RGB8)
    {
        remapped.resize(width * height * 4);
        for (size_t y=0; y<height; ++y)
        {
            for (size_t x=0; x<width; ++x)
            {
                size_t i = (y * width + x);
                remapped[4 * i    ] = reinterpret_cast<const uint8_t*>(pixels)[3 * i    ];
                remapped[4 * i + 1] = reinterpret_cast<const uint8_t*>(pixels)[3 * i + 1];
                remapped[4 * i + 2] = reinterpret_cast<const uint8_t*>(pixels)[3 * i + 2];
                remapped[4 * i + 3] = 255;
            }
        }
        pixels = reinterpret_cast<void*>(remapped.data());
        pixel_type = PIXEL_TYPE::RGBA8;
    }

    switch (image_type)
    {
        case IMAGE_TYPE::REFERENCE:
        {
            assert(pixel_type == PIXEL_TYPE::RGBA8);
            reference_image_width = width;
            reference_image_height = height;
            const auto size = 4 * width * height;
            const auto bytes = sizeof(uint8_t) * size;
            reference_color_buffer = std::vector<uint8_t>(size);
            std::memcpy(reference_color_buffer.data(), pixels, bytes);
            //const auto reference_image = cv::Mat(height, width, CV_8UC4, reference_color_buffer.data());
            reference_image = cv::Mat(height, width, CV_8UC4, reference_color_buffer.data());
            init(reference_image);
            break;
        }
        case IMAGE_TYPE::QUERY:
        {
            assert(pixel_type == PIXEL_TYPE::RGBA8);
            query_image_width = width;
            query_image_height = height;
            const auto size = 4 * width * height;
            const auto bytes = sizeof(uint8_t) * size;
            query_color_buffer = std::vector<uint8_t>(size);
            std::memcpy(query_color_buffer.data(), pixels, bytes);
            query_image = cv::Mat(height, width, CV_8UC4, query_color_buffer.data());
            break;
        }
        case IMAGE_TYPE::DEPTH3D:
        {
            assert(pixel_type == PIXEL_TYPE::F32X3);
            const auto size = 3 * width * height;
            const auto bytes = sizeof(float) * size;
            query_depth3d_buffer = std::vector<float>(size);
            std::memcpy(query_depth3d_buffer.data(), pixels, bytes);
            break;
        }
    }
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
void feature_matcher<Detector, Descriptor, Matcher>::match()
{
    std::vector<cv::KeyPoint> current_keypoints;
    match_result = match_result_t();

    if (!matcher_initialized) return;
    cv::Mat current_descriptors;
    detector->detect(query_image, current_keypoints, cv::noArray());
    descriptor->compute(query_image, current_keypoints, current_descriptors);
    matcher->match(current_descriptors, match_result.matches, cv::noArray());
    match_result.num_ref_descriptors = reference_descriptors.size().height;
    match_result.reference_keypoints = reference_keypoints;
    match_result.query_keypoints = current_keypoints;

    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    //cv::Mat img;
    //cv::drawMatches(query_image, current_keypoints, reference_image, reference_keypoints, match_result.matches, img);
    ////cv::drawMatches(query_image, current_keypoints, query_image, reference_keypoints, match_result.matches, img);
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
    double fy = 0.5 * height / std::tan(0.5 * fovy);
    double cx = (width - 1) / 2.0;
    double cy = (height - 1) / 2.0;
    camera_matrix_data = std::vector<double>{fy, 0, cx, 0, fy, cy, 0, 0, 1};
    camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_data.data());
}

template <typename Detector, typename Descriptor, typename Matcher>
bool feature_matcher<Detector, Descriptor, Matcher>::update_camera(
        std::array<float, 3>& eye,
        std::array<float, 3>& center,
        std::array<float, 3>& up)
{
    std::cout << "old camera:\n" << make_cam_string(eye, center, up) << "\n";

    // calc dist eye to center for later use
    auto distance = cv::norm(cv::Point3f(eye[0], eye[1], eye[2]) - cv::Point3f(center[0], center[1], center[2]));

    auto good_matches = match_result.good_matches(good_match_threshold);
    std::sort(good_matches.begin(), good_matches.end(),
              [](const cv::DMatch& lhs, const cv::DMatch& rhs){return lhs.distance < rhs.distance;});
    std::vector<cv::Point2f> reference_points;
    std::vector<cv::Point2f> query_points;
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
        auto index = 3 * (std::llround(p.y) * query_image_width + std::llround(p.x));
        cv::Point3f coord {query_depth3d_buffer[index], query_depth3d_buffer[index+1], query_depth3d_buffer[index+2]};
        cv::Point3f magicNumber{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        if (coord == magicNumber) // magic value: Pixel has low contribution and should be ignored.
            continue;
        coord.z = -coord.z; // inv z-axis
        query_coords.push_back(coord);
        reference_points_filtered.push_back(reference_points[i]);
    }

    std::cout << "Matches: " << match_result.matches.size() << "\n"
              << "Good matches: " << good_matches.size() << "\n"
              << "Usable coords: " << query_coords.size() << "\n";

    constexpr size_t min_points_for_solvepnp{6};
    if (query_coords.size() < min_points_for_solvepnp)
        return false;
//#define SOLVEPNP_USE_ONLY_MIN_POINTS_FOR_SOLVEPNP
#ifdef SOLVEPNP_USE_ONLY_MIN_POINTS_FOR_SOLVEPNP
    query_coords.resize(6);
    reference_points_filtered.resize(6);
#endif

    // solve
    cv::Mat rotation = cv::Mat(3, 1, CV_64FC1, 0.0);
    cv::Mat translation = cv::Mat(3, 1, CV_64FC1, 0.0);
    try
    {
//#define USE_RANSAC
#ifndef USE_RANSAC
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
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error in solvePnP:\n" << e.what() << std::endl;
        return false;
    }

    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation, rotation_matrix);

    // camera eye
    std::array<double, 3> e{eye[0], eye[1], eye[2]};
    cv::Mat eye_cv = cv::Mat(3, 1, CV_64F, e.data());
    cv::Mat new_eye = -1.0 * rotation_matrix.t() * translation;
    new_eye.at<double>(2) = -new_eye.at<double>(2); // inv z-axis
    eye[0] = static_cast<float>(new_eye.at<double>(0));
    eye[1] = static_cast<float>(new_eye.at<double>(1));
    eye[2] = static_cast<float>(new_eye.at<double>(2));

    // camera up
    std::array<double, 3> u{0.0, 1.0, 0.0};
    cv::Mat default_up = cv::Mat(3, 1, CV_64F, u.data());
    cv::Mat new_up = rotation_matrix.t() * default_up;
    cv::Mat new_up_normalized;
    cv::normalize(new_up, new_up_normalized);
    new_up_normalized.at<double>(2) = -new_up_normalized.at<double>(2); // inv z-axis
    up[0] = static_cast<float>(new_up_normalized.at<double>(0));
    up[1] = static_cast<float>(new_up_normalized.at<double>(1));
    up[2] = static_cast<float>(new_up_normalized.at<double>(2));

    // camera dir
    std::array<double, 3> z{0.0, 0.0, 1.0};
    cv::Mat default_z = cv::Mat(3, 1, CV_64F, z.data());
    cv::Mat new_dir = rotation_matrix.t() * default_z;
    cv::Mat new_dir_normalized;
    cv::normalize(new_dir, new_dir_normalized);
    new_dir_normalized.at<double>(2) = -new_dir_normalized.at<double>(2); // inv z-axis
    
    // camera center
    cv::Mat1d new_center = new_eye + distance * new_dir_normalized;
    center[0] = static_cast<float>(new_center(0));
    center[1] = static_cast<float>(new_center(1));
    center[2] = static_cast<float>(new_center(2));

    std::cout << "new camera:\n" << make_cam_string(eye, center, up) << "\n";

    return true;
}

template <typename Detector, typename Descriptor, typename Matcher>
std::string feature_matcher<Detector, Descriptor, Matcher>::make_cam_string(
        std::array<float, 3>& eye,
        std::array<float, 3>& center,
        std::array<float, 3>& up)
{
    std::stringstream ss;
    ss << "\teye=(" << eye[0] << ", " << eye[1] << ", " << eye[2] << ")"
       << "\n\tcenter=(" << center[0] << ", " << center[1] << ", " << center[2] << ")"
       << "\n\tup=(" << up[0] << ", " << up[1] << ", " << up[2] << ")";
    return ss.str();
}

template <typename Detector, typename Descriptor, typename Matcher>
void feature_matcher<Detector, Descriptor, Matcher>::set_good_match_threshold(float threshold)
{
    good_match_threshold = threshold;
}
