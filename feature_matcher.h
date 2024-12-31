#pragma once

#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/features2d.hpp> // cv::SIFT, cv::ORB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp> // cv::xfeatures2d::SURF
#if HAVE_CUDA
#include <opencv2/cudafeatures2d.hpp> // cv::cuda::ORB
#endif

#include "match_result.h"

namespace detector_type
{
    typedef cv::ORB ORB;
    typedef cv::xfeatures2d::SURF SURF;
#if HAVE_CUDA
    typedef cv::cuda::ORB ORB_GPU;
#endif
}
namespace descriptor_type
{
    typedef cv::ORB ORB;
    typedef cv::SIFT SIFT;
#if HAVE_CUDA
    typedef cv::cuda::ORB ORB_GPU;
#endif
}
namespace matcher_type
{
    typedef cv::BFMatcher BFMatcher;
#if HAVE_CUDA
    typedef cv::cuda::DescriptorMatcher BFMatcher_GPU;
#endif
}

template <typename Detector, typename Descriptor, typename Matcher>
struct feature_matcher
{
    enum PIXEL_TYPE {
        PIXEL_TYPE_UNKNOWN = 0,
        RGBA = 1
    };
    enum DEPTH_TYPE {
        DEPTH_TYPE_UNKNOWN = 0,
        FLOAT = 1,
        FLOAT3 = 2
    };

    feature_matcher();
    void set_reference_image(const std::vector<uint8_t>& data, size_t width, size_t height, PIXEL_TYPE pixel_type);
    void init(const cv::Mat& reference_image);
    void match(const std::vector<uint8_t>& data, size_t width, size_t height, PIXEL_TYPE pixel_type);
    void calibrate(size_t width, size_t height, float fovy, float aspect);
    bool update_camera(const std::vector<float>& depth,
                       const std::vector<float>& depth3D,
                       size_t width,
                       size_t height,
                       DEPTH_TYPE depth_type,
                       DEPTH_TYPE depth3D_type,
                       std::array<float, 3>& eye,
                       std::array<float, 3>& center,
                       std::array<float, 3>& up);

    cv::Ptr<Detector>           detector;
    cv::Ptr<Descriptor>         descriptor;
    cv::Ptr<Matcher>            matcher;
    bool                        matcher_initialized;
    cv::Mat                     reference_descriptors;
    cv::Mat                     camera_matrix;
    std::vector<cv::KeyPoint>   reference_keypoints;
    match_result_t              match_result;
#if HAVE_CUDA
    cv::cuda::GpuMat            gpu_reference_descriptors;
#endif
};

