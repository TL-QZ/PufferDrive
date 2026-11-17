/* Private metadata and canonical scene fingerprint exports for calibration. */

#include <stdint.h>

static void probe_hash_byte(uint64_t hashes[4], unsigned char byte) {
    static const uint64_t primes[4] = {
        UINT64_C(1099511628211), UINT64_C(1099511627791), UINT64_C(1099511627689), UINT64_C(1099511627567),
    };
    for (int lane = 0; lane < 4; lane++) {
        hashes[lane] ^= (uint64_t) (byte + (unsigned char) (lane * 37));
        hashes[lane] *= primes[lane];
        hashes[lane] ^= hashes[lane] >> 29;
    }
}

static void probe_hash_u64(uint64_t hashes[4], uint64_t value) {
    for (int byte_idx = 0; byte_idx < 8; byte_idx++) {
        probe_hash_byte(hashes, (unsigned char) (value >> (byte_idx * 8)));
    }
}

static void probe_hash_i32(uint64_t hashes[4], int value) {
    probe_hash_u64(hashes, (uint64_t) (int64_t) value);
}

static void probe_hash_float(uint64_t hashes[4], float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    probe_hash_u64(hashes, (uint64_t) bits);
}

static void probe_hash_log(uint64_t hashes[4], const Log *log) {
#define PROBE_LOG_FLOAT(name) probe_hash_float(hashes, log->name)
    PROBE_LOG_FLOAT(n); PROBE_LOG_FLOAT(episode_return); PROBE_LOG_FLOAT(episode_length);
    PROBE_LOG_FLOAT(expert_static_car_count); PROBE_LOG_FLOAT(static_car_count); PROBE_LOG_FLOAT(score);
    PROBE_LOG_FLOAT(offroad_rate); PROBE_LOG_FLOAT(collision_rate); PROBE_LOG_FLOAT(red_light_violation_rate);
    PROBE_LOG_FLOAT(num_goals_reached); PROBE_LOG_FLOAT(comfort_violation_count);
    PROBE_LOG_FLOAT(comfort_violation_timestep_count); PROBE_LOG_FLOAT(velocity_progress_sum);
    PROBE_LOG_FLOAT(lane_center_rate); PROBE_LOG_FLOAT(lane_heading_aligned_rate); PROBE_LOG_FLOAT(dnf_rate);
    PROBE_LOG_FLOAT(avg_displacement_error); PROBE_LOG_FLOAT(avg_speed_per_agent);
    PROBE_LOG_FLOAT(at_fault_collision_rate); PROBE_LOG_FLOAT(ttc_within_bound_rate);
    PROBE_LOG_FLOAT(driving_direction_score); PROBE_LOG_FLOAT(speed_limit_compliance);
    PROBE_LOG_FLOAT(making_progress_rate); PROBE_LOG_FLOAT(progress_ratio); PROBE_LOG_FLOAT(comfort_score);
    PROBE_LOG_FLOAT(puffer_score); PROBE_LOG_FLOAT(wrong_way_distance); PROBE_LOG_FLOAT(speed_violation_sum);
    PROBE_LOG_FLOAT(ttc_violations); PROBE_LOG_FLOAT(ttc_samples); PROBE_LOG_FLOAT(multi_lane_time);
    PROBE_LOG_FLOAT(multi_lane_score); PROBE_LOG_FLOAT(total_distance_travelled);
    PROBE_LOG_FLOAT(total_infractions); PROBE_LOG_FLOAT(no_at_fault); PROBE_LOG_FLOAT(no_offroad);
    PROBE_LOG_FLOAT(no_red_light); PROBE_LOG_FLOAT(making_progress); PROBE_LOG_FLOAT(ttc_puffer_rate);
    PROBE_LOG_FLOAT(multiplier); PROBE_LOG_FLOAT(weighted_average); PROBE_LOG_FLOAT(reached_goal_gt);
    PROBE_LOG_FLOAT(reward_collision); PROBE_LOG_FLOAT(reward_offroad); PROBE_LOG_FLOAT(reward_red_light);
    PROBE_LOG_FLOAT(reward_goal); PROBE_LOG_FLOAT(reward_lane_align); PROBE_LOG_FLOAT(reward_lane_center);
    PROBE_LOG_FLOAT(reward_comfort); PROBE_LOG_FLOAT(reward_velocity); PROBE_LOG_FLOAT(reward_timestep);
    PROBE_LOG_FLOAT(reward_reverse); PROBE_LOG_FLOAT(reward_overspeed); PROBE_LOG_FLOAT(reward_ade);
#undef PROBE_LOG_FLOAT
}

static void probe_hash_agent(uint64_t hashes[4], const Agent *agent) {
#define PROBE_AGENT_INT(name) probe_hash_i32(hashes, agent->name)
#define PROBE_AGENT_FLOAT(name) probe_hash_float(hashes, agent->name)
    PROBE_AGENT_INT(id); PROBE_AGENT_INT(type); PROBE_AGENT_INT(trajectory_size);
    PROBE_AGENT_FLOAT(sim_x); PROBE_AGENT_FLOAT(sim_y); PROBE_AGENT_FLOAT(sim_z);
    PROBE_AGENT_FLOAT(sim_heading); PROBE_AGENT_FLOAT(cos_heading); PROBE_AGENT_FLOAT(sin_heading);
    PROBE_AGENT_FLOAT(sim_vx); PROBE_AGENT_FLOAT(sim_vy); PROBE_AGENT_FLOAT(yaw_rate);
    PROBE_AGENT_FLOAT(sim_speed); PROBE_AGENT_FLOAT(sim_speed_signed); PROBE_AGENT_FLOAT(sim_length);
    PROBE_AGENT_FLOAT(sim_width); PROBE_AGENT_FLOAT(sim_height); PROBE_AGENT_FLOAT(radius);
    PROBE_AGENT_FLOAT(prev_x); PROBE_AGENT_FLOAT(prev_y); PROBE_AGENT_FLOAT(prev_cos_heading);
    PROBE_AGENT_FLOAT(prev_sin_heading); PROBE_AGENT_INT(sim_valid);
    PROBE_AGENT_INT(route_length); PROBE_AGENT_INT(route_gt_len); PROBE_AGENT_INT(current_route_idx);
    for (int route_idx = 0; route_idx < agent->route_length; route_idx++) {
        probe_hash_i32(hashes, agent->route[route_idx]);
    }
    for (int metric_idx = 0; metric_idx < NUM_METRICS; metric_idx++) {
        probe_hash_float(hashes, agent->metrics_array[metric_idx]);
    }
    PROBE_AGENT_INT(current_lane_idx); PROBE_AGENT_INT(previous_lane_idx);
    PROBE_AGENT_INT(current_lane_geometry_idx); PROBE_AGENT_INT(reached_goal_this_episode);
    PROBE_AGENT_INT(num_goals_reached); PROBE_AGENT_INT(active_agent); PROBE_AGENT_INT(mark_as_expert);
    PROBE_AGENT_INT(controller); PROBE_AGENT_FLOAT(cumulative_displacement);
    PROBE_AGENT_INT(displacement_sample_count); PROBE_AGENT_FLOAT(distance_since_spawn);
    PROBE_AGENT_FLOAT(seconds_stopped);
    for (int goal_idx = 0; goal_idx < MAX_GOALS; goal_idx++) {
        probe_hash_float(hashes, agent->list_goal_x[goal_idx]);
        probe_hash_float(hashes, agent->list_goal_y[goal_idx]);
        probe_hash_float(hashes, agent->list_goal_z[goal_idx]);
        probe_hash_i32(hashes, agent->list_goal_lane[goal_idx]);
    }
    PROBE_AGENT_FLOAT(current_goal_x); PROBE_AGENT_FLOAT(current_goal_y); PROBE_AGENT_FLOAT(current_goal_z);
    PROBE_AGENT_INT(current_goal_idx); PROBE_AGENT_INT(goal_count);
    PROBE_AGENT_FLOAT(gt_goal_x); PROBE_AGENT_FLOAT(gt_goal_y); PROBE_AGENT_FLOAT(gt_goal_z);
    PROBE_AGENT_INT(stopped); PROBE_AGENT_INT(removed);
    PROBE_AGENT_FLOAT(accel_long); PROBE_AGENT_FLOAT(accel_lat); PROBE_AGENT_FLOAT(jerk_long);
    PROBE_AGENT_FLOAT(jerk_lat); PROBE_AGENT_FLOAT(steering_angle); PROBE_AGENT_FLOAT(wheelbase);
    for (int coefficient_idx = 0; coefficient_idx < NUM_REWARD_COEFS; coefficient_idx++) {
        probe_hash_float(hashes, agent->reward_coefs[coefficient_idx]);
    }
    PROBE_AGENT_INT(phantom_braking_counter); PROBE_AGENT_INT(partner_blindness_counter);
    probe_hash_u64(hashes, (uint64_t) agent->is_blind_partner);
    probe_hash_u64(hashes, (uint64_t) agent->is_phantom_braker);
#undef PROBE_AGENT_INT
#undef PROBE_AGENT_FLOAT
}

static void probe_scene_hash(const Drive *env, uint64_t hashes[4]) {
    hashes[0] = UINT64_C(1469598103934665603);
    hashes[1] = UINT64_C(7809847782465536322);
    hashes[2] = UINT64_C(9650029242287828579);
    hashes[3] = UINT64_C(2870177450012600261);
    probe_hash_i32(hashes, env->num_agents);
    probe_hash_i32(hashes, env->num_controllable_agents);
    probe_hash_i32(hashes, env->active_agent_count);
    probe_hash_i32(hashes, env->num_total_agents);
    probe_hash_i32(hashes, env->static_agent_count);
    probe_hash_i32(hashes, env->expert_static_agent_count);
    probe_hash_i32(hashes, env->timestep);
    probe_hash_i32(hashes, env->init_step);
    probe_hash_i32(hashes, env->scenario_length);
    probe_hash_i32(hashes, env->eval_episode_done);
    probe_hash_u64(hashes, env->episode_seed);
    probe_hash_u64(hashes, env->log_episode_seed);
    probe_hash_u64(hashes, env->init_seed);
    probe_hash_u64(hashes, env->rng_state.s[0]);
    probe_hash_u64(hashes, env->rng_state.s[1]);
    probe_hash_u64(hashes, env->seed_stream_rng.s[0]);
    probe_hash_u64(hashes, env->seed_stream_rng.s[1]);
    probe_hash_u64(hashes, (uint64_t) env->probe_hold_boundary);
    probe_hash_u64(hashes, (uint64_t) env->probe_at_fault_enabled);
    probe_hash_log(hashes, &env->log);
    for (int slot_idx = 0; slot_idx < env->num_controllable_agents; slot_idx++) {
        probe_hash_i32(
            hashes,
            slot_idx < env->active_agent_count ? env->active_agent_indices[slot_idx] : -1);
        probe_hash_u64(hashes, (uint64_t) env->masks[slot_idx]);
        int observation_size = compute_observation_size((Drive *) env);
        for (int feature_idx = 0; feature_idx < observation_size; feature_idx++) {
            probe_hash_float(
                hashes,
                env->observations[(size_t) slot_idx * (size_t) observation_size + (size_t) feature_idx]);
        }
        probe_hash_float(hashes, env->rewards[slot_idx]);
        probe_hash_u64(hashes, (uint64_t) env->terminals[slot_idx]);
        probe_hash_u64(hashes, (uint64_t) env->truncations[slot_idx]);
        if (env->capture_final_observations) {
            probe_hash_u64(hashes, (uint64_t) env->final_observation_valid[slot_idx]);
            for (int feature_idx = 0; feature_idx < observation_size; feature_idx++) {
                probe_hash_float(
                    hashes,
                    env->final_observations[(size_t) slot_idx * (size_t) observation_size + (size_t) feature_idx]);
            }
        }
        if (env->action_type == ACTION_TYPE_CONTINUOUS) {
            float (*action_array)[2] = (float (*)[2]) env->actions;
            probe_hash_float(hashes, action_array[slot_idx][0]);
            probe_hash_float(hashes, action_array[slot_idx][1]);
        } else {
            probe_hash_i32(hashes, ((int *) env->actions)[slot_idx]);
        }
    }
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        probe_hash_agent(hashes, &env->agents[agent_idx]);
    }
    for (int static_idx = 0; static_idx < env->static_agent_count; static_idx++) {
        probe_hash_i32(hashes, env->static_agent_indices[static_idx]);
    }
    for (int expert_idx = 0; expert_idx < env->expert_static_agent_count; expert_idx++) {
        probe_hash_i32(hashes, env->expert_static_agent_indices[expert_idx]);
    }
    for (int scene_agent_idx = 0; scene_agent_idx < env->active_agent_count; scene_agent_idx++) {
        probe_hash_log(hashes, &env->logs[scene_agent_idx]);
    }
    for (int control_idx = 0; control_idx < env->num_traffic_elements; control_idx++) {
        const TrafficControlElement *control = &env->traffic_elements[control_idx];
        probe_hash_i32(hashes, control->type);
        probe_hash_i32(hashes, control->state_size);
        for (int state_idx = 0; state_idx < control->state_size; state_idx++) {
            probe_hash_i32(hashes, control->states[state_idx]);
        }
        probe_hash_float(hashes, control->heading);
        for (int pose_idx = 0; pose_idx < 6; pose_idx++) {
            probe_hash_float(hashes, control->stop_line[pose_idx]);
        }
    }
}

static PyObject *vec_set_probe_mode(PyObject *self __attribute__((unused)), PyObject *args) {
    Py_ssize_t argument_count = PyTuple_Size(args);
    if (argument_count != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_probe_mode requires vec, hold_boundary, and at_fault_enabled");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }
    int hold_boundary = PyObject_IsTrue(PyTuple_GetItem(args, 1));
    int at_fault_enabled = PyObject_IsTrue(PyTuple_GetItem(args, 2));
    if (hold_boundary < 0 || at_fault_enabled < 0) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        Drive *env = vec->envs[env_idx];
        if (env == NULL) {
            PyErr_Format(PyExc_RuntimeError, "probe mode found an uninitialized scene at index %d", env_idx);
            return NULL;
        }
        env->probe_hold_boundary = hold_boundary;
        env->probe_at_fault_enabled = at_fault_enabled;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_get_probe_fingerprints(PyObject *self __attribute__((unused)), PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }
    npy_intp dimensions[2] = {(npy_intp) vec->num_envs, 4};
    PyArrayObject *array = (PyArrayObject *) PyArray_SimpleNew(2, dimensions, NPY_UINT64);
    if (array == NULL) {
        return NULL;
    }
    uint64_t *out = (uint64_t *) PyArray_DATA(array);
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        if (vec->envs[env_idx] == NULL) {
            Py_DECREF(array);
            PyErr_Format(PyExc_RuntimeError, "fingerprint query found an uninitialized scene at index %d", env_idx);
            return NULL;
        }
        probe_scene_hash(vec->envs[env_idx], out + 4 * env_idx);
    }
    return (PyObject *) array;
}

static int probe_set_dict_item(PyObject *dict, const char *name, PyArrayObject *array) {
    if (array == NULL) {
        return -1;
    }
    return PyDict_SetItemString(dict, name, (PyObject *) array);
}

static PyObject *vec_get_probe_metadata(PyObject *self __attribute__((unused)), PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }
    npy_intp row_count = 0;
    int max_traffic_controls = 0;
    int max_traffic_state_size = 0;
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        Drive *env = vec->envs[env_idx];
        if (env == NULL || env->num_controllable_agents < 0 || env->active_agent_count < 0
            || env->active_agent_count > env->num_controllable_agents
            || (env->num_controllable_agents > 0
                && (env->agents == NULL || env->masks == NULL || env->active_agent_indices == NULL))) {
            PyErr_Format(PyExc_RuntimeError, "metadata query found invalid scene buffers at index %d", env_idx);
            return NULL;
        }
        row_count += (npy_intp) env->num_controllable_agents;
        if (env->num_traffic_elements > max_traffic_controls) {
            max_traffic_controls = env->num_traffic_elements;
        }
        for (int control_idx = 0; control_idx < env->num_traffic_elements; control_idx++) {
            if (env->traffic_elements[control_idx].state_size > max_traffic_state_size) {
                max_traffic_state_size = env->traffic_elements[control_idx].state_size;
            }
        }
    }

    npy_intp row_dims[1] = {row_count};
    npy_intp scene_dims[1] = {(npy_intp) vec->num_envs};
    npy_intp xyz_dims[2] = {row_count, 3};
    npy_intp coefficient_dims[2] = {row_count, NUM_REWARD_COEFS};
    npy_intp traffic_dims[2] = {(npy_intp) vec->num_envs, (npy_intp) max_traffic_controls};
    npy_intp traffic_pose_dims[3] = {(npy_intp) vec->num_envs, (npy_intp) max_traffic_controls, 7};
    npy_intp traffic_state_dims[3] = {
        (npy_intp) vec->num_envs, (npy_intp) max_traffic_controls, (npy_intp) max_traffic_state_size};

#define PROBE_ARRAY_1D(name, dtype) (PyArrayObject *) PyArray_SimpleNew(1, row_dims, dtype)
    PyArrayObject *scene_idx = PROBE_ARRAY_1D("scene_idx", NPY_INT32);
    PyArrayObject *agent_id = PROBE_ARRAY_1D("agent_id", NPY_INT32);
    PyArrayObject *active = PROBE_ARRAY_1D("active", NPY_UINT8);
    PyArrayObject *sim_valid = PROBE_ARRAY_1D("sim_valid", NPY_UINT8);
    PyArrayObject *stopped = PROBE_ARRAY_1D("stopped", NPY_UINT8);
    PyArrayObject *removed = PROBE_ARRAY_1D("removed", NPY_UINT8);
    PyArrayObject *is_blind_partner = PROBE_ARRAY_1D("is_blind_partner", NPY_UINT8);
    PyArrayObject *is_phantom_braker = PROBE_ARRAY_1D("is_phantom_braker", NPY_UINT8);
    PyArrayObject *collision = PROBE_ARRAY_1D("collision", NPY_UINT8);
    PyArrayObject *at_fault_collision = PROBE_ARRAY_1D("at_fault_collision", NPY_UINT8);
    PyArrayObject *offroad = PROBE_ARRAY_1D("offroad", NPY_UINT8);
    PyArrayObject *red_light = PROBE_ARRAY_1D("red_light", NPY_UINT8);
    PyArrayObject *reached_goal = PROBE_ARRAY_1D("reached_goal", NPY_UINT8);
    PyArrayObject *num_goals_reached = PROBE_ARRAY_1D("num_goals_reached", NPY_INT32);
    PyArrayObject *current_goal_idx = PROBE_ARRAY_1D("current_goal_idx", NPY_INT32);
    PyArrayObject *goal_count = PROBE_ARRAY_1D("goal_count", NPY_INT32);
    PyArrayObject *goal_lane_idx = PROBE_ARRAY_1D("goal_lane_idx", NPY_INT32);
    PyArrayObject *pose_xyz = (PyArrayObject *) PyArray_SimpleNew(2, xyz_dims, NPY_FLOAT32);
    PyArrayObject *goal_xyz = (PyArrayObject *) PyArray_SimpleNew(2, xyz_dims, NPY_FLOAT32);
    PyArrayObject *heading_rad = PROBE_ARRAY_1D("heading_rad", NPY_FLOAT32);
    PyArrayObject *speed_mps = PROBE_ARRAY_1D("speed_mps", NPY_FLOAT32);
    PyArrayObject *reward_coefficients = (PyArrayObject *) PyArray_SimpleNew(2, coefficient_dims, NPY_FLOAT32);
    PyArrayObject *timestep = (PyArrayObject *) PyArray_SimpleNew(1, scene_dims, NPY_INT32);
    PyArrayObject *episode_seed = (PyArrayObject *) PyArray_SimpleNew(1, scene_dims, NPY_UINT64);
    PyArrayObject *scene_done = (PyArrayObject *) PyArray_SimpleNew(1, scene_dims, NPY_UINT8);
    PyArrayObject *scene_active_agent_count = (PyArrayObject *) PyArray_SimpleNew(1, scene_dims, NPY_INT32);
    PyArrayObject *traffic_control_count = (PyArrayObject *) PyArray_SimpleNew(1, scene_dims, NPY_INT32);
    PyArrayObject *traffic_control_max_state_size
        = (PyArrayObject *) PyArray_SimpleNew(1, scene_dims, NPY_INT32);
    PyArrayObject *traffic_control_type = (PyArrayObject *) PyArray_SimpleNew(2, traffic_dims, NPY_INT32);
    PyArrayObject *traffic_control_state_size = (PyArrayObject *) PyArray_SimpleNew(2, traffic_dims, NPY_INT32);
    PyArrayObject *traffic_control_pose = (PyArrayObject *) PyArray_SimpleNew(3, traffic_pose_dims, NPY_FLOAT32);
    PyArrayObject *traffic_control_state = (PyArrayObject *) PyArray_SimpleNew(3, traffic_state_dims, NPY_INT32);
#undef PROBE_ARRAY_1D

    PyArrayObject *arrays[] = {
        scene_idx, agent_id, active, sim_valid, stopped, removed, is_blind_partner, is_phantom_braker,
        collision, at_fault_collision, offroad, red_light, reached_goal, num_goals_reached,
        current_goal_idx, goal_count, goal_lane_idx, pose_xyz, goal_xyz, heading_rad, speed_mps,
        reward_coefficients, timestep, episode_seed, scene_done, scene_active_agent_count, traffic_control_count,
        traffic_control_max_state_size, traffic_control_type, traffic_control_state_size,
        traffic_control_pose, traffic_control_state,
    };
    const int array_count = (int) (sizeof(arrays) / sizeof(arrays[0]));
    for (int array_idx = 0; array_idx < array_count; array_idx++) {
        if (arrays[array_idx] == NULL) {
            for (int free_idx = 0; free_idx < array_count; free_idx++) {
                Py_XDECREF(arrays[free_idx]);
            }
            return NULL;
        }
    }

    int32_t *scene_idx_data = (int32_t *) PyArray_DATA(scene_idx);
    int32_t *agent_id_data = (int32_t *) PyArray_DATA(agent_id);
    uint8_t *active_data = (uint8_t *) PyArray_DATA(active);
    uint8_t *sim_valid_data = (uint8_t *) PyArray_DATA(sim_valid);
    uint8_t *stopped_data = (uint8_t *) PyArray_DATA(stopped);
    uint8_t *removed_data = (uint8_t *) PyArray_DATA(removed);
    uint8_t *blind_data = (uint8_t *) PyArray_DATA(is_blind_partner);
    uint8_t *phantom_data = (uint8_t *) PyArray_DATA(is_phantom_braker);
    uint8_t *collision_data = (uint8_t *) PyArray_DATA(collision);
    uint8_t *fault_data = (uint8_t *) PyArray_DATA(at_fault_collision);
    uint8_t *offroad_data = (uint8_t *) PyArray_DATA(offroad);
    uint8_t *red_data = (uint8_t *) PyArray_DATA(red_light);
    uint8_t *goal_event_data = (uint8_t *) PyArray_DATA(reached_goal);
    int32_t *goals_reached_data = (int32_t *) PyArray_DATA(num_goals_reached);
    int32_t *current_goal_data = (int32_t *) PyArray_DATA(current_goal_idx);
    int32_t *goal_count_data = (int32_t *) PyArray_DATA(goal_count);
    int32_t *goal_lane_data = (int32_t *) PyArray_DATA(goal_lane_idx);
    float *pose_data = (float *) PyArray_DATA(pose_xyz);
    float *goal_xyz_data = (float *) PyArray_DATA(goal_xyz);
    float *heading_data = (float *) PyArray_DATA(heading_rad);
    float *speed_data = (float *) PyArray_DATA(speed_mps);
    float *coefficient_data = (float *) PyArray_DATA(reward_coefficients);
    int32_t *timestep_data = (int32_t *) PyArray_DATA(timestep);
    uint64_t *episode_seed_data = (uint64_t *) PyArray_DATA(episode_seed);
    uint8_t *scene_done_data = (uint8_t *) PyArray_DATA(scene_done);
    int32_t *scene_active_count_data = (int32_t *) PyArray_DATA(scene_active_agent_count);
    int32_t *traffic_count_data = (int32_t *) PyArray_DATA(traffic_control_count);
    int32_t *traffic_max_size_data = (int32_t *) PyArray_DATA(traffic_control_max_state_size);
    int32_t *traffic_type_data = (int32_t *) PyArray_DATA(traffic_control_type);
    int32_t *traffic_size_data = (int32_t *) PyArray_DATA(traffic_control_state_size);
    float *traffic_pose_data = (float *) PyArray_DATA(traffic_control_pose);
    int32_t *traffic_state_data = (int32_t *) PyArray_DATA(traffic_control_state);
    int row_idx = 0;
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        Drive *env = vec->envs[env_idx];
        timestep_data[env_idx] = env->timestep;
        episode_seed_data[env_idx] = env->episode_seed;
        scene_done_data[env_idx] = (uint8_t) env->eval_episode_done;
        scene_active_count_data[env_idx] = env->active_agent_count;
        traffic_count_data[env_idx] = env->num_traffic_elements;
        int max_state_size = 0;
        for (int control_idx = 0; control_idx < env->num_traffic_elements; control_idx++) {
            if (env->traffic_elements[control_idx].state_size > max_state_size) {
                max_state_size = env->traffic_elements[control_idx].state_size;
            }
        }
        traffic_max_size_data[env_idx] = max_state_size;
        for (int control_idx = 0; control_idx < max_traffic_controls; control_idx++) {
            size_t traffic_row = (size_t) env_idx * (size_t) max_traffic_controls + (size_t) control_idx;
            traffic_type_data[traffic_row] = -1;
            traffic_size_data[traffic_row] = 0;
            for (int pose_idx = 0; pose_idx < 7; pose_idx++) {
                traffic_pose_data[traffic_row * 7 + (size_t) pose_idx] = NAN;
            }
            for (int state_idx = 0; state_idx < max_traffic_state_size; state_idx++) {
                traffic_state_data[traffic_row * (size_t) max_traffic_state_size + (size_t) state_idx] = -1;
            }
            if (control_idx >= env->num_traffic_elements) {
                continue;
            }
            const TrafficControlElement *control = &env->traffic_elements[control_idx];
            traffic_type_data[traffic_row] = control->type;
            traffic_size_data[traffic_row] = control->state_size;
            for (int pose_idx = 0; pose_idx < 6; pose_idx++) {
                traffic_pose_data[traffic_row * 7 + (size_t) pose_idx] = control->stop_line[pose_idx];
            }
            traffic_pose_data[traffic_row * 7 + 6] = control->heading;
            for (int state_idx = 0; state_idx < control->state_size; state_idx++) {
                traffic_state_data[traffic_row * (size_t) max_traffic_state_size + (size_t) state_idx]
                    = control->states[state_idx];
            }
        }
        for (int slot_idx = 0; slot_idx < env->num_controllable_agents; slot_idx++) {
            int is_active_slot = slot_idx < env->active_agent_count;
            int agent_idx = is_active_slot ? env->active_agent_indices[slot_idx] : -1;
            size_t row = (size_t) row_idx;
            scene_idx_data[row_idx] = env_idx;
            agent_id_data[row_idx] = -1;
            active_data[row_idx] = 0;
            sim_valid_data[row_idx] = 0;
            stopped_data[row_idx] = 0;
            removed_data[row_idx] = 0;
            blind_data[row_idx] = 0;
            phantom_data[row_idx] = 0;
            collision_data[row_idx] = 0;
            fault_data[row_idx] = 0;
            offroad_data[row_idx] = 0;
            red_data[row_idx] = 0;
            goal_event_data[row_idx] = 0;
            goals_reached_data[row_idx] = 0;
            current_goal_data[row_idx] = -1;
            goal_count_data[row_idx] = 0;
            goal_lane_data[row_idx] = -1;
            heading_data[row_idx] = NAN;
            speed_data[row_idx] = NAN;
            for (int axis = 0; axis < 3; axis++) {
                pose_data[row * 3 + (size_t) axis] = NAN;
                goal_xyz_data[row * 3 + (size_t) axis] = NAN;
            }
            for (int coefficient_idx = 0; coefficient_idx < NUM_REWARD_COEFS; coefficient_idx++) {
                coefficient_data[row * NUM_REWARD_COEFS + (size_t) coefficient_idx] = NAN;
            }
            if (!is_active_slot) {
                row_idx++;
                continue;
            }
            if (agent_idx < 0 || agent_idx >= env->num_agents) {
                PyErr_Format(PyExc_RuntimeError, "metadata query found invalid agent index %d in scene %d", agent_idx, env_idx);
                goto probe_metadata_error;
            }
            Agent *agent = &env->agents[agent_idx];
            agent_id_data[row_idx] = agent->id;
            active_data[row_idx] = (uint8_t) agent->active_agent;
            sim_valid_data[row_idx] = (uint8_t) agent->sim_valid;
            stopped_data[row_idx] = (uint8_t) agent->stopped;
            removed_data[row_idx] = (uint8_t) agent->removed;
            blind_data[row_idx] = agent->is_blind_partner;
            phantom_data[row_idx] = agent->is_phantom_braker;
            collision_data[row_idx] = (uint8_t) (agent->metrics_array[COLLISION_IDX] > 0.0f);
            fault_data[row_idx] = (uint8_t) (agent->metrics_array[AT_FAULT_COLLISION_IDX] > 0.0f);
            offroad_data[row_idx] = (uint8_t) (agent->metrics_array[OFFROAD_IDX] > 0.0f);
            red_data[row_idx] = (uint8_t) (agent->metrics_array[RED_LIGHT_IDX] > 0.0f);
            goal_event_data[row_idx] = (uint8_t) (agent->metrics_array[REACHED_GOAL_IDX] > 0.0f);
            goals_reached_data[row_idx] = agent->num_goals_reached;
            current_goal_data[row_idx] = agent->current_goal_idx;
            goal_count_data[row_idx] = agent->goal_count;
            pose_data[row * 3] = agent->sim_x;
            pose_data[row * 3 + 1] = agent->sim_y;
            pose_data[row * 3 + 2] = agent->sim_z;
            heading_data[row_idx] = agent->sim_heading;
            speed_data[row_idx] = agent->sim_speed;
            for (int goal_idx = 0; goal_idx < agent->goal_count && goal_idx < MAX_GOALS; goal_idx++) {
                if (goal_idx == agent->current_goal_idx) {
                    goal_xyz_data[row * 3] = agent->list_goal_x[goal_idx];
                    goal_xyz_data[row * 3 + 1] = agent->list_goal_y[goal_idx];
                    goal_xyz_data[row * 3 + 2] = agent->list_goal_z[goal_idx];
                    goal_lane_data[row_idx] = agent->list_goal_lane[goal_idx];
                    break;
                }
            }
            for (int coefficient_idx = 0; coefficient_idx < NUM_REWARD_COEFS; coefficient_idx++) {
                coefficient_data[row * NUM_REWARD_COEFS + (size_t) coefficient_idx]
                    = agent->reward_coefs[coefficient_idx];
            }
            row_idx++;
        }
    }

    PyObject *result = PyDict_New();
    if (result == NULL) {
        goto probe_metadata_error;
    }
#define PROBE_ADD(name, array) \
    do { \
        if (probe_set_dict_item(result, name, array) < 0) { \
            Py_DECREF(result); \
            goto probe_metadata_error; \
        } \
        array = NULL; \
    } while (0)
    PROBE_ADD("scene_idx", scene_idx);
    PROBE_ADD("agent_id", agent_id);
    PROBE_ADD("active", active);
    PROBE_ADD("sim_valid", sim_valid);
    PROBE_ADD("stopped", stopped);
    PROBE_ADD("removed", removed);
    PROBE_ADD("is_blind_partner", is_blind_partner);
    PROBE_ADD("is_phantom_braker", is_phantom_braker);
    PROBE_ADD("collision", collision);
    PROBE_ADD("at_fault_collision", at_fault_collision);
    PROBE_ADD("offroad", offroad);
    PROBE_ADD("red_light", red_light);
    PROBE_ADD("reached_goal", reached_goal);
    PROBE_ADD("num_goals_reached", num_goals_reached);
    PROBE_ADD("current_goal_idx", current_goal_idx);
    PROBE_ADD("goal_count", goal_count);
    PROBE_ADD("goal_lane_idx", goal_lane_idx);
    PROBE_ADD("pose_xyz", pose_xyz);
    PROBE_ADD("goal_xyz", goal_xyz);
    PROBE_ADD("heading_rad", heading_rad);
    PROBE_ADD("speed_mps", speed_mps);
    PROBE_ADD("reward_coefficients", reward_coefficients);
    PROBE_ADD("timestep", timestep);
    PROBE_ADD("episode_seed", episode_seed);
    PROBE_ADD("scene_done", scene_done);
    PROBE_ADD("scene_active_agent_count", scene_active_agent_count);
    PROBE_ADD("traffic_control_count", traffic_control_count);
    PROBE_ADD("traffic_control_max_state_size", traffic_control_max_state_size);
    PROBE_ADD("traffic_control_type", traffic_control_type);
    PROBE_ADD("traffic_control_state_size", traffic_control_state_size);
    PROBE_ADD("traffic_control_pose", traffic_control_pose);
    PROBE_ADD("traffic_control_state", traffic_control_state);
#undef PROBE_ADD
    for (int array_idx = 0; array_idx < array_count; array_idx++) {
        Py_DECREF(arrays[array_idx]);
        arrays[array_idx] = NULL;
    }
    return result;

probe_metadata_error:
    for (int array_idx = 0; array_idx < array_count; array_idx++) {
        Py_XDECREF(arrays[array_idx]);
    }
    return NULL;
}
