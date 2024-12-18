#pragma once

#include "feature_matcher.h"

typedef feature_matcher<detector_type::ORB, descriptor_type::ORB, matcher_type::BFMatcher> matcher_t;

extern "C" {
    matcher_t* create_matcher();
    void destroy_matcher(matcher_t* matcher);
    const char* get_matcher_type();
    const char* get_matcher_description();
}
