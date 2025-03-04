#pragma once

#include <array>
#include <dlfcn.h>
#include <cstdint>
#include <string>
#include <vector>

struct image_transform_estimator
{
  enum PIXEL_TYPE
  {
    RGBA8 = 0,
    RGB8 = 1,
    R8 = 2,
    F32 = 3,
    F32X3 = 4
  };

  enum IMAGE_TYPE {
      REFERENCE = 0,
      QUERY = 1,
      DEPTH3D = 2
  };

  virtual void calibrate(size_t width, size_t height, float fovy, float aspect) = 0;
  virtual void match() = 0;
  virtual void set_image(const void* data,
                         size_t width,
                         size_t height,
                         PIXEL_TYPE pixel_type,
                         IMAGE_TYPE image_type,
                         bool swizzle) = 0;
  virtual bool update_camera(std::array<float, 3>& eye,
                             std::array<float, 3>& center,
                             std::array<float, 3>& up) = 0;
  virtual void set_good_match_threshold(float threshold) = 0;
};
