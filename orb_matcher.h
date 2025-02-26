#pragma once

#include "image_transform_estimator.h"
#include "feature_matcher.h"

typedef feature_matcher<detector_type::ORB, descriptor_type::ORB, matcher_type::BFMatcher> matcher_t;

extern "C" {
    image_transform_estimator* create_estimator();
    void destroy_estimator(image_transform_estimator* estimator);
    const char* get_estimator_type();
    const char* get_estimator_description();
}
