
#include "mock_matcher.h"

template class feature_matcher<int, int, int>;

extern "C" {
    image_transform_estimator* create_estimator()
    {
        return dynamic_cast<image_transform_estimator*>(new matcher_t());
    }

    void destroy_estimator(image_transform_estimator* estimator)
    {
        delete dynamic_cast<image_transform_estimator*>(estimator);
    }

    const char* get_estimator_type()
    {
        return "TEST";
    }

    const char* get_estimator_description()
    {
        return "Detector: int, Descriptor: int, Matcher: int";
    }
}
