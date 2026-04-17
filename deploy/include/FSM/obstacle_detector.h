// Copyright (c) 2026, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <vector>
#include <cmath>

class ObstacleDetector {
public:
    bool detect(const std::vector<float>& lidar) const
    {
        if (lidar.empty()) {
            return false;
        }

        // If all readings are at max distance, sensor is inactive
        if (std::all_of(lidar.begin(), lidar.end(), [](float v) { return v > 4.9f; })) {
            return false;
        }

        const float threshold = 0.7f;
        const auto min_it = std::min_element(lidar.begin(), lidar.end());
        return min_it != lidar.end() && *min_it < threshold;
    }

    int direction(const std::vector<float>& lidar) const
    {
        if (lidar.empty()) {
            return 0;
        }

        const size_t mid = lidar.size() / 2;
        float left_sum = 0.0f;
        float right_sum = 0.0f;

        for (size_t i = 0; i < lidar.size(); ++i) {
            if (i < mid) {
                left_sum += lidar[i];
            } else {
                right_sum += lidar[i];
            }
        }

        if (left_sum < right_sum) {
            return -1;
        }
        if (right_sum < left_sum) {
            return +1;
        }
        return 0;
    }
};

class SafeActionWrapper {
public:
    std::vector<float> process(std::vector<float> action, const std::vector<float>& lidar) const
    {
        // Check if we have valid lidar data
        bool has_valid_lidar = !lidar.empty() && 
            std::any_of(lidar.begin(), lidar.end(), [](float v) { return v < 4.9f; });
        
        bool obstacle_detected = has_valid_lidar && detector_.detect(lidar);
        
        // Without valid lidar, don't apply avoidance (let physics handle it)
        if (!has_valid_lidar) {
            return action;
        }

        // With valid lidar but no obstacle detected, return normal action
        if (!obstacle_detected) {
            return action;
        }

        // OBSTACLE DETECTED: Apply aggressive avoidance
        const int dir = detector_.direction(lidar);
        
        // Strongly reduce forward/lateral motion when obstacle ahead
        for (auto & value : action) {
            value *= 0.2f;
        }

        // Add strong turning correction to steer away
        // Positive dir = turn left (obstacle on right), negative = turn right (obstacle on left)
        for (size_t i = 0; i < action.size(); ++i) {
            if ((i % 2) == 0) {
                action[i] += dir * 0.3f;  // Increased steering magnitude
            }
        }

        return action;
    }

private:
    ObstacleDetector detector_;
};

