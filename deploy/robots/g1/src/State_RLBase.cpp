#include "FSM/State_RLBase.h"
#include "FSM/obstacle_detector.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    const auto lidar = env->observation_manager->get("lidar_scan");
    static SafeActionWrapper safe_action_wrapper;
    action = safe_action_wrapper.process(action, lidar);

    // DEBUG: Log ankle joint actions every 0.5 seconds
    static int debug_counter = 0;
    // if (++debug_counter % 120 == 0) {  // Every 0.5s at 240Hz
    //     //printf("[ACTION_DEBUG] ankle_pitch_L: %.3f, ankle_roll_L: %.3f, ankle_pitch_R: %.3f, ankle_roll_R: %.3f\n",
    //       //     action[10], action[11], action[22], action[23]);
        
    //     // Contact sensor not available in deploy environment
    //     // TODO: Add contact force debugging when sensor access is implemented
        
    //     fflush(stdout);
    // }
    
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
