#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <chrono>

class FSMState : public BaseState
{
public:
    static bool use_joystick;

    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        if(use_joystick)
        {
            auto transitions = param::config["FSM"][state_string]["transitions"];

            if(transitions)
            {
                auto transition_map = transitions.as<std::map<std::string, std::string>>();

                for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
                {
                    std::string target_fsm = it->first;
                    if(!FSMStringMap.right.count(target_fsm))
                    {
                        spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                        continue;
                    }

                    int fsm_id = FSMStringMap.right.at(target_fsm);

                    std::string condition = it->second;
                    unitree::common::dsl::Parser p(condition);
                    auto ast = p.Parse();
                    auto func = unitree::common::dsl::Compile(*ast);
                    registered_checks.emplace_back(
                        std::make_pair(
                            [func]()->bool{ return func(FSMState::lowstate->joystick); },
                            fsm_id
                        )
                    );
                }
            }
        }
        else
        {
            spdlog::info("FSM: joystick disabled, using auto startup transitions.");
            if(state_string == "Passive")
            {
                registered_checks.emplace_back(
                    std::make_pair(
                        [this]()->bool {
                            return std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - _enter_time
                            ).count() > 1.0;
                        },
                        FSMStringMap.right.at("FixStand")
                    )
                );
            }
            else if(state_string == "FixStand")
            {
                registered_checks.emplace_back(
                    std::make_pair(
                        [this]()->bool {
                            return std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - _enter_time
                            ).count() > 3.0;
                        },
                        FSMStringMap.right.at("Velocity")
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void enter() override
    {
        _enter_time = std::chrono::steady_clock::now();
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;

private:
    std::chrono::steady_clock::time_point _enter_time = std::chrono::steady_clock::now();
};