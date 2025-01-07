
#include "mock_matcher.h"

template class feature_matcher<int, int, int>;

extern "C" {
    matcher_t* create_matcher()
    {
        return new matcher_t();
    }

    void destroy_matcher(matcher_t* matcher)
    {
        delete matcher;
    }

    const char* get_matcher_type()
    {
        return "TEST";
    }

    const char* get_matcher_description()
    {
        return "Detector: int, Descriptor: int, Matcher: int";
    }
}
