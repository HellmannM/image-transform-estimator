#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp> // cv::Mat
#include <opencv2/video/tracking.hpp>

#include "ecc.h"

ecc::ecc()
    : reference_image_width{0}
    , reference_image_height{0}
    , query_image_width{0}
    , query_image_height{0}
    , reference_color_buffer{}
    , query_color_buffer{}
//    , query_image{cv::noArray()}
//    , reference_image{cv::noArray()}
    , ecc_value{0.f}
//    , homography_matrix{cv::noArray()}
{}

std::vector<uint8_t> ecc::swizzle_image(
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

void ecc::set_image(
    const void* data,
    size_t width,
    size_t height,
    PIXEL_TYPE pixel_type,
    IMAGE_TYPE image_type,
    bool swizzle)
{
    if (image_type == IMAGE_TYPE::DEPTH3D)
        return;

    const void* pixels{data};
    std::vector<uint8_t> swizzled;
    if (swizzle)
    {
        swizzled = swizzle_image(data, width, height, pixel_type);
        pixels = reinterpret_cast<void*>(swizzled.data());
    }

    //convert to F32
    std::vector<float> remapped;
    switch (pixel_type)
    {
        case PIXEL_TYPE::R8:
        {
            remapped.resize(width * height);
            for (size_t y=0; y<height; ++y)
            {
                for (size_t x=0; x<width; ++x)
                {
                    size_t i = (y * width + x);
                    const auto& pixels_u8 = reinterpret_cast<const uint8_t*>(pixels);
                    remapped[i] = pixels_u8[i] / 255.f;
                }
            }
            pixels = reinterpret_cast<void*>(remapped.data());
            pixel_type = PIXEL_TYPE::F32;
            break;
        }
        case PIXEL_TYPE::RGB8:
        {
            remapped.resize(width * height);
            for (size_t y=0; y<height; ++y)
            {
                for (size_t x=0; x<width; ++x)
                {
                    size_t i = (y * width + x);
                    const auto& pixels_u8 = reinterpret_cast<const uint8_t*>(pixels);
                    remapped[i] =   pixels_u8[3 * i    ];
                                  + pixels_u8[3 * i + 1];
                                  + pixels_u8[3 * i + 2];
                    remapped[i] /= (3 * 255);
                }
            }
            pixels = reinterpret_cast<void*>(remapped.data());
            pixel_type = PIXEL_TYPE::F32;
            break;
        }
        case PIXEL_TYPE::RGBA8:
        {
            remapped.resize(width * height);
            for (size_t y=0; y<height; ++y)
            {
                for (size_t x=0; x<width; ++x)
                {
                    size_t i = (y * width + x);
                    const auto& pixels_u8 = reinterpret_cast<const uint8_t*>(pixels);
                    float val =   pixels_u8[4 * i    ];
                                + pixels_u8[4 * i + 1];
                                + pixels_u8[4 * i + 2];
                    remapped[i] = val * pixels_u8[4 * i + 3] / (4 * 255);
                }
            }
            pixels = reinterpret_cast<void*>(remapped.data());
            pixel_type = PIXEL_TYPE::F32;
            break;
        }
        case PIXEL_TYPE::F32:
            break;
        case PIXEL_TYPE::F32X3:
            std::cerr << "ECC Error: pixel type not supported.\n";
            return;
    }

    assert(pixel_type == PIXEL_TYPE::F32);
    const auto size = width * height;
    const auto bytes = sizeof(float) * size;
    switch (image_type)
    {
        case IMAGE_TYPE::REFERENCE:
        {
            reference_image_width = width;
            reference_image_height = height;
            reference_color_buffer = std::vector<float>(size);
            std::memcpy(reference_color_buffer.data(), pixels, bytes);
            reference_image = cv::Mat(height, width, CV_32FC1, reference_color_buffer.data());
            break;
        }
        case IMAGE_TYPE::QUERY:
        {
            query_image_width = width;
            query_image_height = height;
            query_color_buffer = std::vector<float>(size);
            std::memcpy(query_color_buffer.data(), pixels, bytes);
            query_image = cv::Mat(height, width, CV_32FC1, query_color_buffer.data());
            break;
        }
        case IMAGE_TYPE::DEPTH3D:
            break;
    }
}

void ecc::init(const cv::Mat& reference_image)
{
}
    
void ecc::match()
{
    homography_matrix = cv::Mat::eye(3, 3, CV_32FC1);
//    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001),

    ecc_value = cv::findTransformECC(
        query_image,
        reference_image,
        homography_matrix,
        cv::MOTION_HOMOGRAPHY
    );
//        criteria,
//        cv::noArray(), //input_mask
//        5 //gaussFiltSize
//    );
}

void ecc::calibrate(size_t width, size_t height, float fovy, float aspect)
{
    double fy = 0.5 * height / std::tan(0.5 * fovy);
    double cx = (width - 1) / 2.0;
    double cy = (height - 1) / 2.0;
    camera_matrix_data = std::vector<double>{fy, 0, cx, 0, fy, cy, 0, 0, 1};
    camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_data.data());
}

bool ecc::update_camera(
        std::array<float, 3>& eye,
        std::array<float, 3>& center,
        std::array<float, 3>& up)
{
    std::cout << "old camera:\n" << make_cam_string(eye, center, up) << "\n";

    // calc dist eye to center for later use
    auto distance = cv::norm(cv::Point3f(center[0], center[1], center[2]) - cv::Point3f(eye[0], eye[1], eye[2]));

    // decompose
    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    std::vector<cv::Mat> normals;
    cv::decomposeHomographyMat(homography_matrix, camera_matrix, rotations, translations, normals);
    // check which decomposition to use
    std::array<double, 3> d{center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]};
    cv::Mat dir_cv = cv::Mat(3, 1, CV_64F, d.data());
    cv::Vec3d dir_cv_vec3d(dir_cv.reshape(3).at<cv::Vec3d>());
    auto bestDecompositionIndex = getBestDecompositionIndex(rotations, translations, normals, dir_cv_vec3d);
    if (bestDecompositionIndex < 0)
    {
        std::cerr << "No valid decomposition! Not updating camera.\n";
        return false;
    }
    auto R = rotations[bestDecompositionIndex];
    auto translation = translations[bestDecompositionIndex];

    // camera eye
    std::array<double, 3> e{eye[0], eye[1], eye[2]};
    cv::Mat eye_cv = cv::Mat(3, 1, CV_64F, e.data());
    eye_cv += distance * translation;
    eye[0] = static_cast<float>(eye_cv.at<double>(0));
    eye[1] = static_cast<float>(eye_cv.at<double>(1));
    eye[2] = static_cast<float>(eye_cv.at<double>(2));

    // camera up
    std::array<double, 3> u{up[0], up[1], up[2]};
    cv::Mat up_cv = cv::Mat(3, 1, CV_64F, u.data());
    std::cout << "Original Up Vector: " << up_cv.t() << "\n";
    up_cv = R * up_cv;
    up_cv /= cv::norm(up_cv);
    std::cout << "Transformed Up Vector: " << up_cv.t() << "\n";
    up[0] = static_cast<float>(up_cv.at<double>(0));
    up[1] = static_cast<float>(up_cv.at<double>(1));
    up[2] = static_cast<float>(up_cv.at<double>(2));

    // camera dir
    dir_cv = R * dir_cv;
    dir_cv /= cv::norm(dir_cv);
    // camera center
    cv::Mat1d new_center = eye_cv + distance * dir_cv;
    center[0] = static_cast<float>(new_center(0));
    center[1] = static_cast<float>(new_center(1));
    center[2] = static_cast<float>(new_center(2));

    std::cout << "new camera:\n" << make_cam_string(eye, center, up) << "\n";

    return true;
}

std::string ecc::make_cam_string(
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

void ecc::set_good_match_threshold(float threshold)
{
}

ssize_t ecc::getBestDecompositionIndex(const std::vector<cv::Mat>& rotations,
                                       const std::vector<cv::Mat>& translations,
                                       const std::vector<cv::Mat>& normals,
                                       cv::Vec3d dir)
{
    dir /= cv::norm(dir);

    ssize_t bestDecompositionIndex = -1;
    double maxAlignment = -1;

    for (ssize_t i = 0; i < rotations.size(); i++) {
        const auto& R = rotations[i];
        const auto& t = translations[i];
        cv::Vec3d t_vec3d(t.reshape(3).at<cv::Vec3d>());

        // rotate translation
        cv::Mat t_prime = R * t_vec3d;
        cv::Vec3d t_prime_vec3d(t_prime.reshape(3).at<cv::Vec3d>());
        double dotProduct = dir.dot(t_prime_vec3d);

        // filter solutions:
        // - rotated translation moves forward (R*t dot dir > 0)
        // - normal matches image plane normal most closely (normal dot -dir)
        if (dotProduct >= 0.0) {
            auto normalAlignment = std::abs(normals[i].dot(-dir));
            if (normalAlignment > maxAlignment) {
                maxAlignment = normalAlignment;
                bestDecompositionIndex = i;
            }
        }

//        std::cout << "index="<< i << "\n"
//                  << "R:\n" << R << "\n"
//                  << "t:\n" << t << "\n"
//                  << "t_prime:\n" << t_prime << "\n"
//                  << "dotProduct=" << dotProduct << "\n";
    }
//    std::cout << "best index=" << bestDecompositionIndex << "\n";

    return bestDecompositionIndex;
}
