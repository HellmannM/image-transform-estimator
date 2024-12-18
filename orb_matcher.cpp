#include "orb_matcher.h"

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
        return "ORB";
    }

    const char* get_matcher_description()
    {
        return "Detector: ORB, Descriptor: ORB, Matcher: BFMatcher";
    }
}
