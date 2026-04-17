// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include "FSM/FSMState.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <string>
#include <unordered_map>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(lidar_scan)
{
    int num_rays = 16;
    float max_distance = 5.0f;
    try {
        num_rays = params["num_rays"].as<int>();
    } catch (const std::exception&) {
    }
    try {
        max_distance = params["max_distance"].as<float>();
    } catch (const std::exception&) {
    }

    std::vector<float> scan(num_rays, max_distance);
    if (!env->robot->data.lidar_scan.empty()) {
        scan = env->robot->data.lidar_scan;
    }

    if (scan.size() != static_cast<size_t>(num_rays)) {
        scan.resize(num_rays, max_distance);
    }

    for (auto & value : scan) {
        if (value < 0.0f || std::isnan(value)) {
            value = max_distance;
        }
    }

    return scan;
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    // G1 (and similar): keyboard is non-null; policy thread reads latched W/A/S/D commands.
    // Other deployments use the wireless controller joystick.
    if (FSMState::keyboard)
    {
        using clock = std::chrono::steady_clock;

        // Target command (updated by discrete key events).
        // Default to a small forward velocity so the robot moves in simulation.
        static std::array<float, 3> target = {0.4f, 0.0f, 0.0f};
        // Filtered command actually fed to the policy (smooths steps).
        static std::array<float, 3> filtered = {0.0f, 0.0f, 0.0f};
        static clock::time_point last_cmd_update = clock::now();

        // If no mapped key has been seen recently, decay target back to zero.
        // This keeps behavior "hold/tap to move" instead of infinite latching.
        constexpr float kCommandTimeoutSec = 0.35f;
        // First-order smoothing time constant (seconds).
        constexpr float kCommandTauSec = 0.25f;

        // Body-frame twist matches training (UniformVelocityCommand.vel_command_b): [0]=lin_vel_x forward,
        // [1]=lin_vel_y lateral, [2]=ang_vel_z yaw. Joystick uses ly→vx and -lx→vy; W must command +vx, not +vy.
        // 
        // NOTE: Velocity magnitude must be sufficient for policy to generate forward motion.
        // Training ranges: lin_vel_x=[-1.0, 2.0], lin_vel_y=[-1.0, 1.0], ang_vel_z=[-1.0, 1.0]
        // Test values if oscillation occurs:
        //   - 0.5 m/s for steady walking
        //   - 1.0 m/s for normal walking
        //   - 1.5 m/s for faster walking
        static const std::unordered_map<std::string, std::vector<float>> key_commands = {
            {"w", {0.4f, 0.0f, 0.0f}},   // Further reduced from 0.6 to 0.4 for stability
            {"s", {0.0f, 0.0f, 0.0f}},
            {"a", {0.0f, 0.0f, 0.2f}},   // Further reduced turn speed
            {"d", {0.0f, 0.0f, -0.2f}},  // Further reduced turn speed
        };

        std::string key = FSMState::keyboard->key();
        if (key.size() == 1) {
            key[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(key[0])));
        }
        const auto found = key_commands.find(key);
        if (found != key_commands.end()) {
            target = {found->second[0], found->second[1], found->second[2]};
            last_cmd_update = clock::now();
        }

        const float dt = env->step_dt;
        const float alpha = std::clamp(dt / (kCommandTauSec + dt), 0.0f, 1.0f);

        // NOTE: Command timeout removed to enable continuous motion without automatic reset.
        // Commands now persist until a new key is pressed. The first-order smoothing filter
        // (kCommandTauSec = 0.25s) still provides gentle acceleration/deceleration.

        for (int i = 0; i < 3; ++i) {
            filtered[i] = filtered[i] + alpha * (target[i] - filtered[i]);
        }

        obs[0] = std::clamp(filtered[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
        obs[1] = std::clamp(filtered[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
        obs[2] = std::clamp(filtered[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
        
        // DEBUG: Log velocity commands every 0.5 seconds (~240 steps at 500Hz)
        // static int debug_counter = 0;
        // if (++debug_counter % 240 == 0) {
        //     printf("[VEL_CMD] target={%.3f, %.3f, %.3f} filtered={%.3f, %.3f, %.3f} obs={%.3f, %.3f, %.3f}\n",
        //            target[0], target[1], target[2],
        //            filtered[0], filtered[1], filtered[2],
        //            obs[0], obs[1], obs[2]);
        //     printf("[VEL_RANGES] x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]\n",
        //            cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>(),
        //            cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>(),
        //            cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
        //     fflush(stdout);
        // }
        
        return obs;
    }

    auto & joystick = env->robot->data.joystick;

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

}
}