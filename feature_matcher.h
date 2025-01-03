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
        RGBA = 1,
        FLOAT3 = 2
    };

    enum IMAGE_TYPE {
        IMAGE_TYPE_UNKNOWN = 0,
        REFERENCE = 1,
        QUERY = 2,
        DEPTH3D = 3
    };

    feature_matcher();
    void set_image(const void* data,
                   size_t width,
                   size_t height,
                   PIXEL_TYPE pixel_type,
                   IMAGE_TYPE image_type,
                   bool swizzle);
    void init(const cv::Mat& reference_image);
    void match(const cv::Mat& query_image);
    void calibrate(size_t width, size_t height, float fovy, float aspect);
    bool update_camera(std::array<float, 3>& eye,
                       std::array<float, 3>& center,
                       std::array<float, 3>& up);
    std::vector<uint8_t> swizzle(const void* data, size_t width, size_t height, PIXEL_TYPE pixel_type);

    cv::Ptr<Detector>           detector;
    cv::Ptr<Descriptor>         descriptor;
    cv::Ptr<Matcher>            matcher;
    bool                        matcher_initialized;
    cv::Mat                     reference_descriptors;
    cv::Mat                     camera_matrix;
    std::vector<cv::KeyPoint>   reference_keypoints;
    match_result_t              match_result;
    std::vector<float>          depth3d_buffer;
    size_t                      depth3d_width;
    size_t                      depth3d_height;
#if HAVE_CUDA
    cv::cuda::GpuMat            gpu_reference_descriptors;
#endif
};

