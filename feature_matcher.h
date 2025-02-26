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

#include "image_transform_estimator.h"
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
class feature_matcher : public image_transform_estimator
{
    using PIXEL_TYPE = image_transform_estimator::PIXEL_TYPE;
    using IMAGE_TYPE = image_transform_estimator::IMAGE_TYPE;
public:
    feature_matcher();
    virtual void calibrate(size_t width, size_t height, float fovy, float aspect) override;
    virtual void match() override;
    virtual void set_image(const void* data,
                           size_t width,
                           size_t height,
                           PIXEL_TYPE pixel_type,
                           IMAGE_TYPE image_type,
                           bool swizzle) override;
    virtual bool update_camera(std::array<float, 3>& eye,
                               std::array<float, 3>& center,
                               std::array<float, 3>& up) override;
    virtual void set_good_match_threshold(float threshold) override;

private:
    void init(const cv::Mat& reference_image);
    std::vector<uint8_t> swizzle_image(const void* data, size_t width, size_t height, PIXEL_TYPE pixel_type);
    std::string make_cam_string(std::array<float, 3>& eye,
                                std::array<float, 3>& center,
                                std::array<float, 3>& up);

    cv::Ptr<Detector>           detector;
    cv::Ptr<Descriptor>         descriptor;
    cv::Ptr<Matcher>            matcher;
    bool                        matcher_initialized;
    cv::Mat                     reference_descriptors;
    std::vector<double>         camera_matrix_data;
    cv::Mat                     camera_matrix;
    std::vector<cv::KeyPoint>   reference_keypoints;
    match_result_t              match_result;
    size_t                      reference_image_width;
    size_t                      reference_image_height;
    size_t                      query_image_width;
    size_t                      query_image_height;
    std::vector<uint8_t>        reference_color_buffer;
    std::vector<uint8_t>        query_color_buffer;
    std::vector<float>          query_depth3d_buffer;
    cv::Mat                     query_image;
    cv::Mat                     reference_image;
#if HAVE_CUDA
    cv::cuda::GpuMat            gpu_reference_descriptors;
#endif
    float                       good_match_threshold;
};

