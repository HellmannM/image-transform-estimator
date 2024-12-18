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
    const std::vector<uint8_t>& data, int width, int height, PIXEL_TYPE pixel_type)
{
    if (pixel_type != PIXEL_TYPE::RGBA)
    {
        std::cerr << "Feature matcher: Unsupported pixel type.\n";
        return;
    }

    #define SHUFFLE
    #ifdef SHUFFLE
    int bpp = 4;
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
    const std::vector<uint8_t>& data, int width, int height, PIXEL_TYPE pixel_type)
{
    if (pixel_type != PIXEL_TYPE::RGBA)
    {
        std::cerr << "Feature matcher: Unsupported pixel type.\n";
        return;
    }
    #define SHUFFLE2
    #ifdef SHUFFLE2
    int bpp = 4;
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

template<>
void feature_matcher<cv::cuda::ORB, cv::cuda::ORB, cv::cuda::DescriptorMatcher>::match(const cv::Mat& current_image)
{
    match_result = match_result_t();
    std::vector<cv::KeyPoint> current_keypoints;
    cv::cuda::GpuMat gpu_current_descriptors;
    cv::cuda::GpuMat gpu_current_image_color(current_image);
    cv::cuda::GpuMat gpu_current_image;
    cv::cuda::cvtColor(gpu_current_image_color, gpu_current_image, cv::COLOR_RGBA2GRAY);

    detector->detectAndCompute(gpu_current_image, cv::noArray(), current_keypoints, gpu_current_descriptors);
    if (!matcher_initialized) return;
    matcher->match(gpu_current_descriptors, match_result.matches);

    match_result.num_ref_descriptors = gpu_reference_descriptors.size().height;
    match_result.reference_keypoints = reference_keypoints;
    match_result.query_keypoints = current_keypoints;
}
#endif
    
