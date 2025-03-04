#pragma once

#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp> // cv::Mat

#include "image_transform_estimator.h"

class ecc : public image_transform_estimator
{
    using PIXEL_TYPE = image_transform_estimator::PIXEL_TYPE;
    using IMAGE_TYPE = image_transform_estimator::IMAGE_TYPE;
public:
    ecc();
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

    size_t                      reference_image_width;
    size_t                      reference_image_height;
    size_t                      query_image_width;
    size_t                      query_image_height;
    std::vector<float>          reference_color_buffer;
    std::vector<float>          query_color_buffer;
    cv::Mat                     query_image;
    cv::Mat                     reference_image;
    std::vector<double>         camera_matrix_data;
    cv::Mat                     camera_matrix;
    float                       ecc_value;
    cv::Mat                     homography_matrix;
};

