#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "match_result.h"

// "smart" comparator: compare match_ratio if significantly different,
// otherwise compare good_distance (if match_ratio is similar).
bool match_result_t::operator<(const match_result_t &rhs) const
{
    if (match_ratio() + match_ratio_compare_offset < rhs.match_ratio())
        return true;
    else if (match_ratio() < rhs.match_ratio())
        return good_matches_distance() < rhs.good_matches_distance();
    else
        return false;
}

// "smart" comparator: compare match_ratio if significantly different,
// otherwise compare good_distance (if match_ratio is similar).
bool match_result_t::operator>(const match_result_t &rhs) const
{
    if (match_ratio() + match_ratio_compare_offset > rhs.match_ratio())
        return true;
    else if (match_ratio() > rhs.match_ratio())
        return good_matches_distance() > rhs.good_matches_distance();
    else
        return false;
}

float match_result_t::match_ratio() const
{
    if (num_ref_descriptors == 0)
        return -1;
    return (float)matches.size() / num_ref_descriptors;
}

float match_result_t::distance() const
{
    if (matches.empty())
        return std::numeric_limits<float>::max();
    float dist = 0.f;
    for (auto &m : matches)
    {
        dist += m.distance;
    }
    return dist;
}

std::vector<cv::DMatch> match_result_t::good_matches(float threshold) const
{
    std::vector<cv::DMatch> gm;
    for (auto &m : matches)
    {
        if (m.distance < threshold)
            gm.push_back(m);
    }
    return gm;
}

float match_result_t::good_matches_ratio(float threshold) const
{
    if (num_ref_descriptors == 0)
        return -1;
    return (float)good_matches(threshold).size() / num_ref_descriptors;
}

float match_result_t::good_matches_distance(float threshold) const
{
    float dist = 0.f;
    for (auto &m : good_matches(threshold))
    {
        dist += m.distance;
    }
    return dist;
}
