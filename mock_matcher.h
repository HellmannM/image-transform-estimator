#pragma once

#include <array>
#include <cstdint>
#include <iostream>

#include "image_transform_estimator.h"

template <typename Detector, typename Descriptor, typename Matcher>
struct feature_matcher : public image_transform_estimator
{
    feature_matcher() = default;

    virtual void calibrate(size_t width, size_t height, float fovy, float aspect) override
    {
        std::cout << "feature_matcher::calibrate:"
                << "\n\twidth=" << width
                << "\n\theight=" << height
                << "\n\tfovy=" << fovy
                << "\n\taspect=" << aspect
                << std::endl;
    }

    virtual void match() override
    {
        std::cout << "feature_matcher::match" << std::endl;
    }

    virtual void set_image(const void* data,
                   size_t width,
                   size_t height,
                   PIXEL_TYPE pixel_type,
                   IMAGE_TYPE image_type,
                   bool swizzle) override
    {
        std::cout << "feature_matcher::set_image:"
                << "\n\tdata=" << data
                << "\n\twidth=" << width
                << "\n\theight=" << height
                << "\n\tpixel_type=" << (int)pixel_type
                << "\n\timage_type=" << (int)image_type
                << "\n\tswizzle=" << std::boolalpha << swizzle
                << std::endl;
    }

    virtual bool update_camera(std::array<float, 3>& eye,
                               std::array<float, 3>& center,
                               std::array<float, 3>& up) override
    {
        std::cout << "feature_matcher::update_camera:"
                << "\n\teye=[" << eye[0] << ", " << eye[1] << ", " << eye[2] << "]"
                << "\n\tcenter=[" << center[0] << ", " << center[1] << ", " << center[2] << "]"
                << "\n\teye=[" << up[0] << ", " << up[1] << ", " << up[2] << "]"
                << std::endl;
        return true;
    }

    virtual void set_good_match_threshold(float threshold) override
    {
        std::cout << "feature_matcher::set_good_match_threshold: " << threshold << "\n";
    }
};

typedef feature_matcher<int, int, int> matcher_t;

extern "C" {
    image_transform_estimator* create_estimator();
    void destroy_estimator(image_transform_estimator* estimator);
    const char* get_estimator_type();
    const char* get_estimator_description();
}