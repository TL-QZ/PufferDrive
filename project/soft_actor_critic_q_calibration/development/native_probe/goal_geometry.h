/* Read exact ordered route goals and motion endpoints without changing scene hashes. */

static PyObject *vec_get_goal_geometry(PyObject *self __attribute__((unused)), PyObject *args) {
    PyObject *handle;
    if (!PyArg_ParseTuple(args, "O:vec_get_goal_geometry", &handle)) {
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }

    npy_intp row_count = 0;
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        Drive *env = vec->envs[env_idx];
        if (env == NULL || env->num_controllable_agents < 0 || env->active_agent_count < 0
            || env->active_agent_count > env->num_controllable_agents
            || (env->num_controllable_agents > 0
                && (env->agents == NULL || env->active_agent_indices == NULL))) {
            PyErr_Format(PyExc_RuntimeError, "goal geometry query found invalid scene buffers at index %d", env_idx);
            return NULL;
        }
        row_count += (npy_intp) env->num_controllable_agents;
    }

    npy_intp row_dims[1] = {row_count};
    npy_intp previous_dims[2] = {row_count, 2};
    npy_intp pose_dims[2] = {row_count, 3};
    npy_intp goals_dims[3] = {row_count, MAX_GOALS, 3};
    PyArrayObject *agent_ids = (PyArrayObject *) PyArray_SimpleNew(1, row_dims, NPY_INT32);
    PyArrayObject *current_goal_indices = (PyArrayObject *) PyArray_SimpleNew(1, row_dims, NPY_INT32);
    PyArrayObject *goal_counts = (PyArrayObject *) PyArray_SimpleNew(1, row_dims, NPY_INT32);
    PyArrayObject *resolved_goal_indices = (PyArrayObject *) PyArray_SimpleNew(1, row_dims, NPY_INT32);
    PyArrayObject *fulfilled_goal_counts = (PyArrayObject *) PyArray_SimpleNew(1, row_dims, NPY_INT32);
    PyArrayObject *goal_radii = (PyArrayObject *) PyArray_SimpleNew(1, row_dims, NPY_FLOAT32);
    PyArrayObject *next_goal_distances = (PyArrayObject *) PyArray_SimpleNew(1, row_dims, NPY_FLOAT32);
    PyArrayObject *previous_xy = (PyArrayObject *) PyArray_SimpleNew(2, previous_dims, NPY_FLOAT32);
    PyArrayObject *pose_xyz = (PyArrayObject *) PyArray_SimpleNew(2, pose_dims, NPY_FLOAT32);
    PyArrayObject *route_goals_xyz = (PyArrayObject *) PyArray_SimpleNew(3, goals_dims, NPY_FLOAT32);
    PyArrayObject *arrays[] = {
        agent_ids, current_goal_indices, goal_counts, resolved_goal_indices,
        fulfilled_goal_counts, goal_radii, next_goal_distances,
        previous_xy, pose_xyz, route_goals_xyz,
    };
    for (size_t array_idx = 0; array_idx < sizeof(arrays) / sizeof(arrays[0]); array_idx++) {
        if (arrays[array_idx] == NULL) {
            for (size_t cleanup_idx = 0; cleanup_idx < sizeof(arrays) / sizeof(arrays[0]); cleanup_idx++) {
                Py_XDECREF(arrays[cleanup_idx]);
            }
            return PyErr_NoMemory();
        }
    }

    int32_t *agent_id_data = (int32_t *) PyArray_DATA(agent_ids);
    int32_t *current_goal_data = (int32_t *) PyArray_DATA(current_goal_indices);
    int32_t *goal_count_data = (int32_t *) PyArray_DATA(goal_counts);
    int32_t *resolved_goal_data = (int32_t *) PyArray_DATA(resolved_goal_indices);
    int32_t *fulfilled_goal_count_data = (int32_t *) PyArray_DATA(fulfilled_goal_counts);
    float *goal_radius_data = (float *) PyArray_DATA(goal_radii);
    float *next_goal_distance_data = (float *) PyArray_DATA(next_goal_distances);
    float *previous_xy_data = (float *) PyArray_DATA(previous_xy);
    float *pose_data = (float *) PyArray_DATA(pose_xyz);
    float *route_goal_data = (float *) PyArray_DATA(route_goals_xyz);
    size_t row = 0;
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        Drive *env = vec->envs[env_idx];
        for (int slot_idx = 0; slot_idx < env->num_controllable_agents; slot_idx++, row++) {
            int active_slot = slot_idx < env->active_agent_count;
            int agent_idx = active_slot ? env->active_agent_indices[slot_idx] : -1;
            agent_id_data[row] = -1;
            current_goal_data[row] = -1;
            goal_count_data[row] = 0;
            resolved_goal_data[row] = -1;
            fulfilled_goal_count_data[row] = 0;
            goal_radius_data[row] = NAN;
            next_goal_distance_data[row] = NAN;
            for (int axis = 0; axis < 2; axis++) {
                previous_xy_data[row * 2 + (size_t) axis] = NAN;
            }
            for (int axis = 0; axis < 3; axis++) {
                pose_data[row * 3 + (size_t) axis] = NAN;
            }
            for (int goal_idx = 0; goal_idx < MAX_GOALS; goal_idx++) {
                for (int axis = 0; axis < 3; axis++) {
                    route_goal_data[(row * MAX_GOALS + (size_t) goal_idx) * 3 + (size_t) axis] = NAN;
                }
            }
            if (!active_slot) {
                continue;
            }
            if (agent_idx < 0 || agent_idx >= env->num_agents) {
                PyErr_Format(PyExc_RuntimeError, "goal geometry query found invalid agent index %d in scene %d", agent_idx, env_idx);
                goto goal_geometry_error;
            }
            Agent *agent = &env->agents[agent_idx];
            if (agent->goal_count < 0 || agent->goal_count > MAX_GOALS
                || agent->current_goal_idx < 0 || agent->current_goal_idx > agent->goal_count) {
                PyErr_Format(PyExc_RuntimeError, "goal geometry query found invalid route indices for agent %d", agent_idx);
                goto goal_geometry_error;
            }
            agent_id_data[row] = agent->id;
            current_goal_data[row] = agent->current_goal_idx;
            goal_count_data[row] = agent->goal_count;
            goal_radius_data[row] = agent->reward_coefs[REWARD_COEF_GOAL_RADIUS];
            previous_xy_data[row * 2] = agent->prev_x;
            previous_xy_data[row * 2 + 1] = agent->prev_y;
            pose_data[row * 3] = agent->sim_x;
            pose_data[row * 3 + 1] = agent->sim_y;
            pose_data[row * 3 + 2] = agent->sim_z;
            for (int goal_idx = 0; goal_idx < agent->goal_count; goal_idx++) {
                route_goal_data[(row * MAX_GOALS + (size_t) goal_idx) * 3] = agent->list_goal_x[goal_idx];
                route_goal_data[(row * MAX_GOALS + (size_t) goal_idx) * 3 + 1] = agent->list_goal_y[goal_idx];
                route_goal_data[(row * MAX_GOALS + (size_t) goal_idx) * 3 + 2] = agent->list_goal_z[goal_idx];
            }
            int resolved_goal_idx = agent->current_goal_idx;
            while (resolved_goal_idx < agent->goal_count) {
                float distance_to_goal = compute_point_to_segment_distance(
                    agent->list_goal_x[resolved_goal_idx],
                    agent->list_goal_y[resolved_goal_idx],
                    agent->prev_x,
                    agent->prev_y,
                    agent->sim_x,
                    agent->sim_y);
                float goal_z_distance = fabsf(agent->sim_z - agent->list_goal_z[resolved_goal_idx]);
                if (distance_to_goal >= agent->reward_coefs[REWARD_COEF_GOAL_RADIUS]
                    || goal_z_distance >= Z_BUFFER) {
                    break;
                }
                resolved_goal_idx++;
                fulfilled_goal_count_data[row]++;
            }
            resolved_goal_data[row] = resolved_goal_idx;
            if (resolved_goal_idx < agent->goal_count) {
                float delta_x = agent->list_goal_x[resolved_goal_idx] - agent->sim_x;
                float delta_y = agent->list_goal_y[resolved_goal_idx] - agent->sim_y;
                next_goal_distance_data[row] = sqrtf(delta_x * delta_x + delta_y * delta_y);
            }
        }
    }

    PyObject *result = Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
        "agent_id", agent_ids,
        "current_goal_idx", current_goal_indices,
        "goal_count", goal_counts,
        "resolved_goal_idx", resolved_goal_indices,
        "fulfilled_goal_count", fulfilled_goal_counts,
        "goal_radius_m", goal_radii,
        "next_goal_distance_m", next_goal_distances,
        "prev_xy", previous_xy,
        "pose_xyz", pose_xyz,
        "route_goals_xyz", route_goals_xyz);
    for (size_t array_idx = 0; array_idx < sizeof(arrays) / sizeof(arrays[0]); array_idx++) {
        Py_DECREF(arrays[array_idx]);
    }
    return result;

goal_geometry_error:
    for (size_t array_idx = 0; array_idx < sizeof(arrays) / sizeof(arrays[0]); array_idx++) {
        Py_DECREF(arrays[array_idx]);
    }
    return NULL;
}
