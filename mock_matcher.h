#pragma once

#include <array>
#include <cstdint>
#include <iostream>

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

    feature_matcher() = default;

    virtual void calibrate(size_t width, size_t height, float fovy, float aspect)
    {
        std::cout << "feature_matcher::calibrate:"
                << "\n\twidth=" << width
                << "\n\theight=" << height
                << "\n\tfovy=" << fovy
                << "\n\taspect=" << aspect
                << std::endl;
    }

    virtual void match()
    {
        std::cout << "feature_matcher::match" << std::endl;
    }

    virtual void set_image(const void* data,
                   size_t width,
                   size_t height,
                   PIXEL_TYPE pixel_type,
                   IMAGE_TYPE image_type,
                   bool swizzle)
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
                               std::array<float, 3>& up)
    {
        std::cout << "feature_matcher::update_camera:"
                << "\n\teye=[" << eye[0] << ", " << eye[1] << ", " << eye[2] << "]"
                << "\n\tcenter=[" << center[0] << ", " << center[1] << ", " << center[2] << "]"
                << "\n\teye=[" << up[0] << ", " << up[1] << ", " << up[2] << "]"
                << std::endl;
        return true;
    }
};

typedef feature_matcher<int, int, int> matcher_t;

extern "C" {
    matcher_t* create_matcher();
    void destroy_matcher(matcher_t* matcher);
    const char* get_matcher_type();
    const char* get_matcher_description();
}