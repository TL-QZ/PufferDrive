/* Safe probe-only extension of a verified gigaflow scene cutoff. */

typedef struct {
    int scene_idx;
    int control_idx;
    int *replacement_states;
} ProbeTrafficExtension;

static int probe_traffic_cycle_period(const int *states, int state_count) {
    if (state_count <= 1) {
        return state_count;
    }
    if ((size_t) state_count > SIZE_MAX / sizeof(int)) {
        return -3;
    }
    int *prefix_lengths = (int *) malloc((size_t) state_count * sizeof(int));
    if (prefix_lengths == NULL) {
        return -1;
    }
    prefix_lengths[0] = 0;
    for (int state_idx = 1; state_idx < state_count; state_idx++) {
        int prefix_length = prefix_lengths[state_idx - 1];
        int fallback_count = 0;
        while (prefix_length > 0 && states[state_idx] != states[prefix_length]) {
            if (fallback_count >= state_count) {
                free(prefix_lengths);
                return -2;
            }
            fallback_count++;
            prefix_length = prefix_lengths[prefix_length - 1];
        }
        if (states[state_idx] == states[prefix_length]) {
            prefix_length++;
        }
        prefix_lengths[state_idx] = prefix_length;
    }
    int period = state_count - prefix_lengths[state_count - 1];
    if (period > state_count / 2) {
        period = state_count;
    }
    for (int state_idx = period; state_idx < state_count; state_idx++) {
        if (states[state_idx] != states[state_idx - period]) {
            period = state_count;
            break;
        }
    }
    free(prefix_lengths);
    return period;
}

static PyObject *vec_start_probe_horizon(PyObject *self __attribute__((unused)), PyObject *args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_start_probe_horizon requires vec and steps");
        return NULL;
    }
    VecEnv *vec = unpack_vecenv(args);
    if (vec == NULL) {
        return NULL;
    }
    PyObject *steps_obj = PyTuple_GetItem(args, 1);
    if (!PyLong_Check(steps_obj) || PyBool_Check(steps_obj)) {
        PyErr_SetString(PyExc_TypeError, "probe horizon steps must be a positive integer");
        return NULL;
    }
    long steps_long = PyLong_AsLong(steps_obj);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (steps_long <= 0 || steps_long > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "probe horizon steps must be in [1, INT_MAX]");
        return NULL;
    }
    if (vec->num_envs <= 0 || vec->envs == NULL) {
        PyErr_SetString(PyExc_ValueError, "probe horizon requires at least one initialized scene");
        return NULL;
    }

    int original_scenario_length = -1;
    int current_timestep = -1;
    size_t extension_count = 0;
    for (int scene_idx = 0; scene_idx < vec->num_envs; scene_idx++) {
        Drive *env = vec->envs[scene_idx];
        if (env == NULL) {
            PyErr_Format(PyExc_RuntimeError, "probe horizon found an uninitialized scene at index %d", scene_idx);
            return NULL;
        }
        if (env->simulation_mode != SIMULATION_MODE_GIGAFLOW) {
            PyErr_SetString(PyExc_ValueError, "probe horizon extension supports gigaflow scenes only");
            return NULL;
        }
        if (!env->probe_hold_boundary || env->probe_horizon_started) {
            PyErr_Format(PyExc_RuntimeError, "scene %d is not ready for its first probe horizon", scene_idx);
            return NULL;
        }
        if (env->scenario_length <= 0 || env->timestep < 0 || env->timestep >= env->scenario_length
            || env->eval_episode_done) {
            PyErr_Format(
                PyExc_ValueError,
                "scene %d is already at or past its original cutoff (timestep=%d, scenario_length=%d)",
                scene_idx,
                env->timestep,
                env->scenario_length);
            return NULL;
        }
        if ((long) env->timestep + steps_long >= INT_MAX) {
            PyErr_SetString(PyExc_OverflowError, "probe horizon cutoff must be less than INT_MAX");
            return NULL;
        }
        if (env->num_traffic_elements < 0
            || (env->num_traffic_elements > 0 && env->traffic_elements == NULL)) {
            PyErr_Format(PyExc_RuntimeError, "scene %d has an invalid traffic-control table", scene_idx);
            return NULL;
        }
        if (scene_idx == 0) {
            original_scenario_length = env->scenario_length;
            current_timestep = env->timestep;
        } else if (env->scenario_length != original_scenario_length || env->timestep != current_timestep) {
            PyErr_SetString(PyExc_ValueError, "probe scene copies do not share one scenario cutoff and timestep");
            return NULL;
        }

        for (int control_idx = 0; control_idx < env->num_traffic_elements; control_idx++) {
            TrafficControlElement *control = &env->traffic_elements[control_idx];
            if (control->state_size < 0 || (control->state_size > 0 && control->states == NULL)) {
                PyErr_Format(
                    PyExc_RuntimeError,
                    "scene %d traffic control %d has an invalid state buffer",
                    scene_idx,
                    control_idx);
                return NULL;
            }
            if ((size_t) control->state_size > SIZE_MAX / sizeof(int)) {
                PyErr_Format(
                    PyExc_OverflowError,
                    "scene %d traffic control %d state buffer is too large",
                    scene_idx,
                    control_idx);
                return NULL;
            }
            if (control->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT) {
                continue;
            }
            if (control->states == NULL || control->state_size != env->scenario_length) {
                PyErr_Format(
                    PyExc_RuntimeError,
                    "scene %d traffic light %d has %d states for a %d-step gigaflow scene",
                    scene_idx,
                    control_idx,
                    control->state_size,
                    env->scenario_length);
                return NULL;
            }
            if ((long) env->timestep + steps_long > control->state_size) {
                if (extension_count == SIZE_MAX) {
                    PyErr_SetString(PyExc_OverflowError, "too many traffic controls to extend");
                    return NULL;
                }
                extension_count++;
            }
        }
    }

    uint64_t reference_hashes[4] = {0, 0, 0, 0};
    for (int scene_idx = 0; scene_idx < vec->num_envs; scene_idx++) {
        uint64_t hashes[4];
        probe_scene_hash(vec->envs[scene_idx], hashes);
        if (scene_idx == 0) {
            memcpy(reference_hashes, hashes, sizeof(reference_hashes));
            continue;
        }
        if (memcmp(reference_hashes, hashes, sizeof(reference_hashes)) != 0) {
            PyErr_SetString(PyExc_AssertionError, "probe scene fingerprints differ before horizon extension");
            return NULL;
        }
    }

    ProbeTrafficExtension *extensions = NULL;
    if (extension_count > 0) {
        if (extension_count > SIZE_MAX / sizeof(ProbeTrafficExtension)) {
            PyErr_SetString(PyExc_OverflowError, "traffic horizon extension plan is too large");
            return NULL;
        }
        extensions = (ProbeTrafficExtension *) calloc(extension_count, sizeof(ProbeTrafficExtension));
        if (extensions == NULL) {
            return PyErr_NoMemory();
        }
    }

    const int new_scenario_length = current_timestep + (int) steps_long;
    size_t extension_idx = 0;
    for (int scene_idx = 0; scene_idx < vec->num_envs; scene_idx++) {
        Drive *env = vec->envs[scene_idx];
        for (int control_idx = 0; control_idx < env->num_traffic_elements; control_idx++) {
            TrafficControlElement *control = &env->traffic_elements[control_idx];
            if (control->type != TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT
                || new_scenario_length <= control->state_size) {
                continue;
            }
            if ((size_t) new_scenario_length > SIZE_MAX / sizeof(int)) {
                for (size_t free_idx = 0; free_idx < extension_idx; free_idx++) {
                    free(extensions[free_idx].replacement_states);
                }
                free(extensions);
                PyErr_SetString(PyExc_OverflowError, "traffic horizon state buffer is too large");
                return NULL;
            }
            int period = probe_traffic_cycle_period(control->states, control->state_size);
            if (period < 0) {
                for (size_t free_idx = 0; free_idx < extension_idx; free_idx++) {
                    free(extensions[free_idx].replacement_states);
                }
                free(extensions);
                if (period == -2) {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "scene %d traffic light %d has an invalid state prefix table",
                        scene_idx,
                        control_idx);
                    return NULL;
                }
                if (period == -3) {
                    PyErr_SetString(PyExc_OverflowError, "traffic state prefix table is too large");
                    return NULL;
                }
                return PyErr_NoMemory();
            }
            if (period >= control->state_size) {
                for (size_t free_idx = 0; free_idx < extension_idx; free_idx++) {
                    free(extensions[free_idx].replacement_states);
                }
                free(extensions);
                PyErr_Format(
                    PyExc_ValueError,
                    "scene %d traffic light %d has no fully observed repeating phase in %d states",
                    scene_idx,
                    control_idx,
                    control->state_size);
                return NULL;
            }
            int *replacement_states = (int *) malloc((size_t) new_scenario_length * sizeof(int));
            if (replacement_states == NULL) {
                for (size_t free_idx = 0; free_idx < extension_idx; free_idx++) {
                    free(extensions[free_idx].replacement_states);
                }
                free(extensions);
                return PyErr_NoMemory();
            }
            memcpy(
                replacement_states,
                control->states,
                (size_t) control->state_size * sizeof(int));
            for (int state_idx = control->state_size; state_idx < new_scenario_length; state_idx++) {
                replacement_states[state_idx] = control->states[state_idx % period];
            }
            extensions[extension_idx].scene_idx = scene_idx;
            extensions[extension_idx].control_idx = control_idx;
            extensions[extension_idx].replacement_states = replacement_states;
            extension_idx++;
        }
    }

    for (size_t plan_idx = 0; plan_idx < extension_idx; plan_idx++) {
        ProbeTrafficExtension *extension = &extensions[plan_idx];
        TrafficControlElement *control
            = &vec->envs[extension->scene_idx]->traffic_elements[extension->control_idx];
        int *previous_states = control->states;
        control->states = extension->replacement_states;
        control->state_size = new_scenario_length;
        extension->replacement_states = NULL;
        free(previous_states);
    }
    free(extensions);

    for (int scene_idx = 0; scene_idx < vec->num_envs; scene_idx++) {
        Drive *env = vec->envs[scene_idx];
        env->scenario_length = new_scenario_length;
        env->compute_eval_metrics = 1;
        env->probe_at_fault_enabled = 1;
        env->probe_horizon_started = 1;
    }
    Py_RETURN_NONE;
}
