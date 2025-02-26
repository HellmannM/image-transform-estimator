#include "surf_matcher.h"

extern "C" {
    image_transform_estimator* create_estimator()
    {
        return dynamic_cast<image_transform_estimator*>(new matcher_t());
    }

    void destroy_estimator(image_transform_estimator* estimator)
    {
        delete dynamic_cast<matcher_t*>(estimator);
    }

    const char* get_estimator_type()
    {
        return "SURF";
    }

    const char* get_estimator_description()
    {
        return "Detector: SURF, Descriptor: SIFT, Matcher: BFMatcher";
    }
}
