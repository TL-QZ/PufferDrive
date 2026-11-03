static PyObject *vec_get_diagnostic(PyObject *self __attribute__((unused)), PyObject *args) {
    enum {
        DIAG_SCENE_IDX,
        DIAG_SLOT_IDX,
        DIAG_AGENT_ID,
        DIAG_EPISODE_SEED,
        DIAG_TIMESTEP,
        DIAG_MASK_PRE_ACTION,
        DIAG_EPISODE_ACTIVE,
        DIAG_ACTIVE_AGENT,
        DIAG_SIM_VALID,
        DIAG_STOPPED,
        DIAG_REMOVED,
        DIAG_CURRENT_LANE_IDX,
        DIAG_CURRENT_GOAL_IDX,
        DIAG_GOAL_COUNT,
        DIAG_CURRENT_GOAL_LANE_IDX,
        DIAG_POSITION_XYZ,
        DIAG_GOAL_XYZ,
        DIAG_GOAL_DISTANCE_XY_METERS,
        DIAG_GRAPH_DISTANCE_METERS,
        DIAG_GRAPH_DISTANCE_STATUS,
        DIAG_GRAPH_REACHABLE,
        DIAG_HEADING_RAD,
        DIAG_SPEED_MPS,
        DIAG_NUM_GOALS_REACHED,
        DIAG_REACHED_GOAL_METRIC,
        DIAG_REWARD_COEFFICIENTS,
        DIAG_FIELD_COUNT
    };

    static const char *field_names[DIAG_FIELD_COUNT] = {
        "scene_idx",
        "slot_idx",
        "agent_id",
        "episode_seed",
        "timestep",
        "mask_pre_action",
        "episode_active",
        "active_agent",
        "sim_valid",
        "stopped",
        "removed",
        "current_lane_idx",
        "current_goal_idx",
        "goal_count",
        "current_goal_lane_idx",
        "position_xyz_meters",
        "goal_xyz_meters",
        "goal_distance_xy_meters",
        "graph_distance_meters",
        "graph_distance_status",
        "graph_reachable",
        "heading_rad",
        "speed_mps",
        "num_goals_reached",
        "reached_goal_metric",
        "reward_coefficients"
    };

    VecEnv *vec = unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }

    npy_intp row_count = 0;
    for (int scene_idx = 0; scene_idx < vec->num_envs; scene_idx++) {
        Drive *env = vec->envs[scene_idx];
        if (env == NULL) {
            PyErr_Format(PyExc_RuntimeError, "diagnostic query found an uninitialized scene at index %d", scene_idx);
            return NULL;
        }
        if (env->num_controllable_agents < 0 || env->active_agent_count < 0
            || env->active_agent_count > env->num_controllable_agents
            || (env->num_controllable_agents > 0 && (env->agents == NULL || env->masks == NULL))
            || (env->active_agent_count > 0 && env->active_agent_indices == NULL)) {
            PyErr_Format(PyExc_RuntimeError, "diagnostic query found invalid active-agent buffers in scene %d", scene_idx);
            return NULL;
        }
        if (row_count > NPY_MAX_INTP - (npy_intp) env->num_controllable_agents) {
            PyErr_SetString(PyExc_OverflowError, "diagnostic row count exceeds NumPy index range");
            return NULL;
        }
        row_count += (npy_intp) env->num_controllable_agents;
    }

    npy_intp vector_dims[1] = {row_count};
    npy_intp position_dims[2] = {row_count, 3};
    npy_intp coefficient_dims[2] = {row_count, NUM_REWARD_COEFS};
    PyArrayObject *arrays[DIAG_FIELD_COUNT] = {NULL};
    arrays[DIAG_SCENE_IDX] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_SLOT_IDX] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_AGENT_ID] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_EPISODE_SEED] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT64);
    arrays[DIAG_TIMESTEP] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_MASK_PRE_ACTION] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_EPISODE_ACTIVE] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_ACTIVE_AGENT] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_SIM_VALID] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_STOPPED] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_REMOVED] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_CURRENT_LANE_IDX] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_CURRENT_GOAL_IDX] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_GOAL_COUNT] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_CURRENT_GOAL_LANE_IDX] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_POSITION_XYZ] = (PyArrayObject *) PyArray_SimpleNew(2, position_dims, NPY_FLOAT32);
    arrays[DIAG_GOAL_XYZ] = (PyArrayObject *) PyArray_SimpleNew(2, position_dims, NPY_FLOAT32);
    arrays[DIAG_GOAL_DISTANCE_XY_METERS] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_FLOAT32);
    arrays[DIAG_GRAPH_DISTANCE_METERS] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_FLOAT32);
    arrays[DIAG_GRAPH_DISTANCE_STATUS] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_GRAPH_REACHABLE] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_UINT8);
    arrays[DIAG_HEADING_RAD] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_FLOAT32);
    arrays[DIAG_SPEED_MPS] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_FLOAT32);
    arrays[DIAG_NUM_GOALS_REACHED] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_INT32);
    arrays[DIAG_REACHED_GOAL_METRIC] = (PyArrayObject *) PyArray_SimpleNew(1, vector_dims, NPY_FLOAT32);
    arrays[DIAG_REWARD_COEFFICIENTS] = (PyArrayObject *) PyArray_SimpleNew(2, coefficient_dims, NPY_FLOAT32);
    for (int field_idx = 0; field_idx < DIAG_FIELD_COUNT; field_idx++) {
        if (arrays[field_idx] == NULL) {
            for (int allocated_idx = 0; allocated_idx < DIAG_FIELD_COUNT; allocated_idx++) {
                Py_XDECREF(arrays[allocated_idx]);
            }
            return NULL;
        }
    }

    int row_idx = 0;
    for (int scene_idx = 0; scene_idx < vec->num_envs; scene_idx++) {
        Drive *env = vec->envs[scene_idx];
        for (int slot_idx = 0; slot_idx < env->num_controllable_agents; slot_idx++) {
            int is_active_slot = slot_idx < env->active_agent_count;
            int agent_idx = is_active_slot ? env->active_agent_indices[slot_idx] : -1;
            if (is_active_slot && (agent_idx < 0 || agent_idx >= env->num_agents)) {
                PyErr_Format(
                    PyExc_RuntimeError,
                    "diagnostic query found invalid agent index %d in scene %d slot %d",
                    agent_idx,
                    scene_idx,
                    slot_idx);
                goto diagnostic_error;
            }

            ((int32_t *) PyArray_DATA(arrays[DIAG_SCENE_IDX]))[row_idx] = (int32_t) scene_idx;
            ((int32_t *) PyArray_DATA(arrays[DIAG_SLOT_IDX]))[row_idx] = (int32_t) slot_idx;
            ((int32_t *) PyArray_DATA(arrays[DIAG_AGENT_ID]))[row_idx] = -1;
            ((uint64_t *) PyArray_DATA(arrays[DIAG_EPISODE_SEED]))[row_idx] = (uint64_t) env->episode_seed;
            ((int32_t *) PyArray_DATA(arrays[DIAG_TIMESTEP]))[row_idx] = (int32_t) env->timestep;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_MASK_PRE_ACTION]))[row_idx] = env->masks[slot_idx];
            ((uint8_t *) PyArray_DATA(arrays[DIAG_EPISODE_ACTIVE]))[row_idx] = 0;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_ACTIVE_AGENT]))[row_idx] = 0;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_SIM_VALID]))[row_idx] = 0;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_STOPPED]))[row_idx] = 0;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_REMOVED]))[row_idx] = 0;
            ((int32_t *) PyArray_DATA(arrays[DIAG_CURRENT_LANE_IDX]))[row_idx] = -1;
            ((int32_t *) PyArray_DATA(arrays[DIAG_CURRENT_GOAL_IDX]))[row_idx] = -1;
            ((int32_t *) PyArray_DATA(arrays[DIAG_GOAL_COUNT]))[row_idx] = 0;
            ((int32_t *) PyArray_DATA(arrays[DIAG_CURRENT_GOAL_LANE_IDX]))[row_idx] = -1;
            *((float *) PyArray_GETPTR2(arrays[DIAG_POSITION_XYZ], row_idx, 0)) = NAN;
            *((float *) PyArray_GETPTR2(arrays[DIAG_POSITION_XYZ], row_idx, 1)) = NAN;
            *((float *) PyArray_GETPTR2(arrays[DIAG_POSITION_XYZ], row_idx, 2)) = NAN;
            *((float *) PyArray_GETPTR2(arrays[DIAG_GOAL_XYZ], row_idx, 0)) = NAN;
            *((float *) PyArray_GETPTR2(arrays[DIAG_GOAL_XYZ], row_idx, 1)) = NAN;
            *((float *) PyArray_GETPTR2(arrays[DIAG_GOAL_XYZ], row_idx, 2)) = NAN;
            ((float *) PyArray_DATA(arrays[DIAG_GOAL_DISTANCE_XY_METERS]))[row_idx] = NAN;
            ((float *) PyArray_DATA(arrays[DIAG_GRAPH_DISTANCE_METERS]))[row_idx] = NAN;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_GRAPH_DISTANCE_STATUS]))[row_idx] = 6;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_GRAPH_REACHABLE]))[row_idx] = 0;
            ((float *) PyArray_DATA(arrays[DIAG_HEADING_RAD]))[row_idx] = NAN;
            ((float *) PyArray_DATA(arrays[DIAG_SPEED_MPS]))[row_idx] = NAN;
            ((int32_t *) PyArray_DATA(arrays[DIAG_NUM_GOALS_REACHED]))[row_idx] = 0;
            ((float *) PyArray_DATA(arrays[DIAG_REACHED_GOAL_METRIC]))[row_idx] = NAN;
            float *reward_coefficients
                = (float *) PyArray_GETPTR2(arrays[DIAG_REWARD_COEFFICIENTS], row_idx, 0);
            for (int coefficient_idx = 0; coefficient_idx < NUM_REWARD_COEFS; coefficient_idx++) {
                reward_coefficients[coefficient_idx] = NAN;
            }

            if (!is_active_slot) {
                row_idx++;
                continue;
            }

            Agent *agent = &env->agents[agent_idx];
            int goal_is_active = agent->current_goal_idx >= 0
                && agent->current_goal_idx < agent->goal_count
                && agent->current_goal_idx < env->num_goals;
            int goal_lane_idx = goal_is_active ? agent->list_goal_lane[agent->current_goal_idx] : -1;
            float *position_xyz = (float *) PyArray_GETPTR2(arrays[DIAG_POSITION_XYZ], row_idx, 0);
            float *goal_xyz = (float *) PyArray_GETPTR2(arrays[DIAG_GOAL_XYZ], row_idx, 0);

            ((int32_t *) PyArray_DATA(arrays[DIAG_AGENT_ID]))[row_idx] = (int32_t) agent->id;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_EPISODE_ACTIVE]))[row_idx]
                = (uint8_t) (!env->eval_episode_done && agent->active_agent && !agent->stopped
                             && !agent->removed && agent->sim_valid);
            ((uint8_t *) PyArray_DATA(arrays[DIAG_ACTIVE_AGENT]))[row_idx] = (uint8_t) agent->active_agent;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_SIM_VALID]))[row_idx] = (uint8_t) agent->sim_valid;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_STOPPED]))[row_idx] = (uint8_t) agent->stopped;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_REMOVED]))[row_idx] = (uint8_t) agent->removed;
            ((int32_t *) PyArray_DATA(arrays[DIAG_CURRENT_LANE_IDX]))[row_idx]
                = (int32_t) agent->current_lane_idx;
            ((int32_t *) PyArray_DATA(arrays[DIAG_CURRENT_GOAL_IDX]))[row_idx]
                = (int32_t) agent->current_goal_idx;
            ((int32_t *) PyArray_DATA(arrays[DIAG_GOAL_COUNT]))[row_idx] = (int32_t) agent->goal_count;
            ((int32_t *) PyArray_DATA(arrays[DIAG_CURRENT_GOAL_LANE_IDX]))[row_idx] = (int32_t) goal_lane_idx;

            position_xyz[0] = agent->sim_x;
            position_xyz[1] = agent->sim_y;
            position_xyz[2] = agent->sim_z;
            float goal_distance_xy_meters = NAN;
            if (goal_is_active) {
                goal_xyz[0] = agent->list_goal_x[agent->current_goal_idx];
                goal_xyz[1] = agent->list_goal_y[agent->current_goal_idx];
                goal_xyz[2] = agent->list_goal_z[agent->current_goal_idx];
                float goal_delta_x = goal_xyz[0] - agent->sim_x;
                float goal_delta_y = goal_xyz[1] - agent->sim_y;
                goal_distance_xy_meters = sqrtf(goal_delta_x * goal_delta_x + goal_delta_y * goal_delta_y);
            } else {
                goal_xyz[0] = NAN;
                goal_xyz[1] = NAN;
                goal_xyz[2] = NAN;
            }
            ((float *) PyArray_DATA(arrays[DIAG_GOAL_DISTANCE_XY_METERS]))[row_idx] = goal_distance_xy_meters;

            uint8_t graph_distance_status = 0;
            float graph_distance_meters = NAN;
            int current_lane_idx = agent->current_lane_idx;
            if (!goal_is_active) {
                graph_distance_status = 0;
            } else if (goal_lane_idx < 0 || goal_lane_idx >= env->num_road_elements
                || current_lane_idx < 0 || current_lane_idx >= env->num_road_elements) {
                graph_distance_status = 1;
            } else if (env->lane_graph.lane_to_graph_idx == NULL || env->lane_graph.distances == NULL
                || env->lane_graph.n_lanes <= 0) {
                graph_distance_status = 2;
            } else {
                int current_graph_idx = env->lane_graph.lane_to_graph_idx[current_lane_idx];
                int goal_graph_idx = env->lane_graph.lane_to_graph_idx[goal_lane_idx];
                if (current_graph_idx < 0 || current_graph_idx >= env->lane_graph.n_lanes
                    || goal_graph_idx < 0 || goal_graph_idx >= env->lane_graph.n_lanes) {
                    graph_distance_status = 3;
                } else {
                    graph_distance_meters
                        = env->lane_graph.distances[current_graph_idx * env->lane_graph.n_lanes + goal_graph_idx];
                    graph_distance_status
                        = (isfinite(graph_distance_meters) && graph_distance_meters >= 0.0f) ? 5 : 4;
                }
            }
            ((float *) PyArray_DATA(arrays[DIAG_GRAPH_DISTANCE_METERS]))[row_idx] = graph_distance_meters;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_GRAPH_DISTANCE_STATUS]))[row_idx] = graph_distance_status;
            ((uint8_t *) PyArray_DATA(arrays[DIAG_GRAPH_REACHABLE]))[row_idx]
                = (uint8_t) (graph_distance_status == 5);
            ((float *) PyArray_DATA(arrays[DIAG_HEADING_RAD]))[row_idx] = agent->sim_heading;
            ((float *) PyArray_DATA(arrays[DIAG_SPEED_MPS]))[row_idx] = agent->sim_speed;
            ((int32_t *) PyArray_DATA(arrays[DIAG_NUM_GOALS_REACHED]))[row_idx]
                = (int32_t) agent->num_goals_reached;
            ((float *) PyArray_DATA(arrays[DIAG_REACHED_GOAL_METRIC]))[row_idx]
                = agent->metrics_array[REACHED_GOAL_IDX];
            reward_coefficients
                = (float *) PyArray_GETPTR2(arrays[DIAG_REWARD_COEFFICIENTS], row_idx, 0);
            for (int coefficient_idx = 0; coefficient_idx < NUM_REWARD_COEFS; coefficient_idx++) {
                reward_coefficients[coefficient_idx] = agent->reward_coefs[coefficient_idx];
            }
            row_idx++;
        }
    }

    PyObject *result = PyDict_New();
    if (result == NULL) {
        goto diagnostic_error;
    }
    for (int field_idx = 0; field_idx < DIAG_FIELD_COUNT; field_idx++) {
        if (PyDict_SetItemString(result, field_names[field_idx], (PyObject *) arrays[field_idx]) < 0) {
            Py_DECREF(result);
            goto diagnostic_error;
        }
        Py_DECREF(arrays[field_idx]);
        arrays[field_idx] = NULL;
    }
    return result;

diagnostic_error:
    for (int field_idx = 0; field_idx < DIAG_FIELD_COUNT; field_idx++) {
        Py_XDECREF(arrays[field_idx]);
    }
    return NULL;
}
