#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

struct match_result_t
{
    // "smart" comparator: compare match_ratio if significantly different, otherwise compare good_distance (if match_ratio is similar).
    bool operator<(const match_result_t& rhs) const;
    // "smart" comparator: compare match_ratio if significantly different, otherwise compare good_distance (if match_ratio is similar).
    bool operator>(const match_result_t& rhs) const;
    float match_ratio() const;
    float distance() const;
    std::vector<cv::DMatch> good_matches(float threshold = good_match_threshold) const;
    float good_matches_ratio(float threshold = good_match_threshold) const;
    float good_matches_distance(float threshold = good_match_threshold) const;

    static constexpr float match_ratio_compare_offset {0.005f};
    static constexpr float good_match_threshold {30.f};

    uint32_t num_ref_descriptors;
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> reference_keypoints;
    std::vector<cv::KeyPoint> query_keypoints;
};
