#ifndef PUFFERLIB_OCEAN_DRIVE_MAP_DATA_H
#define PUFFERLIB_OCEAN_DRIVE_MAP_DATA_H

// ========================================
// Grid Map Functions
// ========================================

static int get_grid_index(Drive *env, float x1, float y1) {
    if (env->grid_map->top_left_x >= env->grid_map->bottom_right_x
        || env->grid_map->bottom_right_y >= env->grid_map->top_left_y) {
        return -1;
    }
    float rel_x = x1 - env->grid_map->top_left_x;
    float rel_y = y1 - env->grid_map->bottom_right_y;
    int grid_x = (int) (rel_x / GRID_CELL_SIZE);
    int grid_y = (int) (rel_y / GRID_CELL_SIZE);
    if (grid_x < 0 || grid_x >= env->grid_map->grid_cols || grid_y < 0 || grid_y >= env->grid_map->grid_rows) {
        return -1;
    }
    return (grid_y * env->grid_map->grid_cols) + grid_x;
}

static void add_entity_to_grid(
    Drive *env,
    int grid_index,
    int entity_idx,
    int geometry_idx,
    int valid_for_obs,
    int *cell_entities_insert_index) {
    if (grid_index == -1) {
        return;
    }

    int count = cell_entities_insert_index[grid_index];
    if (count >= env->grid_map->cell_entities_count[grid_index]) {
        printf(
            "Error: Exceeded precomputed entity count for grid cell %d. Current count: %d, Max count(Precomputed): "
            "%d\n",
            grid_index,
            count,
            env->grid_map->cell_entities_count[grid_index]);
        return;
    }

    env->grid_map->cells[grid_index][count].entity_idx = entity_idx;
    env->grid_map->cells[grid_index][count].geometry_idx = geometry_idx;
    env->grid_map->cells[grid_index][count].valid_for_obs = valid_for_obs;
    cell_entities_insert_index[grid_index] = count + 1;
}

static int init_grid_map(Drive *env) {
    env->grid_map = (GridMap *) calloc(1, sizeof(GridMap));

    float top_left_x = 0.0f, top_left_y = 0.0f, bottom_right_x = 0.0f, bottom_right_y = 0.0f;
    bool first_valid_point = false;
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road_grid_candidate(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_size; j++) {
            if (element->x[j] == INVALID_POSITION || element->y[j] == INVALID_POSITION) {
                continue;
            }
            if (!first_valid_point) {
                top_left_x = bottom_right_x = element->x[j];
                top_left_y = bottom_right_y = element->y[j];
                first_valid_point = true;
                continue;
            }
            if (element->x[j] < top_left_x) {
                top_left_x = element->x[j];
            }
            if (element->x[j] > bottom_right_x) {
                bottom_right_x = element->x[j];
            }
            if (element->y[j] > top_left_y) {
                top_left_y = element->y[j];
            }
            if (element->y[j] < bottom_right_y) {
                bottom_right_y = element->y[j];
            }
        }
    }
    env->grid_map->top_left_x = top_left_x;
    env->grid_map->top_left_y = top_left_y;
    env->grid_map->bottom_right_x = bottom_right_x;
    env->grid_map->bottom_right_y = bottom_right_y;

    float grid_width = bottom_right_x - top_left_x;
    float grid_height = top_left_y - bottom_right_y;
    if (!first_valid_point || !isfinite(grid_width) || !isfinite(grid_height)) {
        fprintf(stderr, "[ERROR] -> Map has no valid road geometry extent for grid\n");
        return 1;
    }
    env->grid_map->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_map->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    long long grid_cell_count_wide = (long long) env->grid_map->grid_cols * (long long) env->grid_map->grid_rows;
    if (env->grid_map->grid_cols < 1 || env->grid_map->grid_rows < 1 || grid_cell_count_wide > MAX_GRID_CELL_COUNT) {
        fprintf(
            stderr,
            "[ERROR] -> Invalid grid dimensions %d x %d from map extent %.1f x %.1f\n",
            env->grid_map->grid_cols,
            env->grid_map->grid_rows,
            grid_width,
            grid_height);
        return 1;
    }
    int grid_cell_count = (int) grid_cell_count_wide;
    env->grid_map->cells = (GridMapEntity **) calloc(grid_cell_count, sizeof(GridMapEntity *));
    env->grid_map->cell_entities_count = (int *) calloc(grid_cell_count, sizeof(int));
    // First pass to count entities in each grid cell
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road_grid_candidate(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        for (int j = 0; j < element->segment_size - 1; j++) {
            float x_center = (element->x[j] + element->x[j + 1]) / 2;
            float y_center = (element->y[j] + element->y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1) {
                continue;
            }
            env->grid_map->cell_entities_count[grid_index]++;
        }
    }
    // Allocate grid cells based on counts
    int *cell_entities_insert_index = (int *) calloc(grid_cell_count, sizeof(int));
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        int count = env->grid_map->cell_entities_count[grid_index];
        env->grid_map->total_entities += count;
        env->grid_map->cells[grid_index] = (GridMapEntity *) calloc(count, sizeof(GridMapEntity));
    }
    // Track which grid cells have drivable lanes
    bool *drivable_grid_seen = (bool *) calloc(grid_cell_count, sizeof(bool));
    for (int i = 0; i < env->num_road_elements; i++) {
        if (!is_road_grid_candidate(env->road_elements[i].type)) {
            continue;
        }
        RoadMapElement *element = &env->road_elements[i];
        int obs_stride = 1;
        if (is_road_lane(element->type)) {
            obs_stride = env->obs_lane_stride;
        } else if (is_road_edge(element->type)) {
            obs_stride = env->obs_boundary_stride;
        }
        int last_kept_idx = 0;
        for (int j = 0; j < element->segment_size - 1; j++) {
            // Keep a point every obs_stride points, plus wherever heading deviates enough
            // since the last kept point (densifies curves/intersections)
            int valid_for_obs = 1;
            if (obs_stride > 1 && j > 0) {
                float heading_dev = fabsf(normalize_heading(element->headings[j] - element->headings[last_kept_idx]));
                valid_for_obs = j - last_kept_idx >= obs_stride || heading_dev > OBS_STRIDE_HEADING_THRESHOLD;
            }
            if (valid_for_obs) {
                last_kept_idx = j;
            }
            float x_center = (element->x[j] + element->x[j + 1]) / 2;
            float y_center = (element->y[j] + element->y[j + 1]) / 2;
            int grid_index = get_grid_index(env, x_center, y_center);
            if (grid_index == -1) {
                continue;
            }
            add_entity_to_grid(env, grid_index, i, j, valid_for_obs, cell_entities_insert_index);
            if (is_drivable_road_lane(element->type) && !drivable_grid_seen[grid_index]) {
                drivable_grid_seen[grid_index] = true;
                env->grid_map->num_drivable_grid_cell++;
            }
        }
    }
    // Create a compact array of drivable grid cell indices for quick access
    env->grid_map->grid_index_drivable = (int *) malloc(env->grid_map->num_drivable_grid_cell * sizeof(int));
    int drivable_idx = 0;
    for (int i = 0; i < grid_cell_count; i++) {
        if (drivable_grid_seen[i]) {
            env->grid_map->grid_index_drivable[drivable_idx++] = i;
        }
    }
    free(drivable_grid_seen);
    free(cell_entities_insert_index);
    return 0;
}

static void init_neighbor_offsets(Drive *env) {
    int vr = env->grid_map->vision_range;
    env->neighbor_offsets = (int *) calloc(vr * vr * 2, sizeof(int));
    // Spiral pattern generation
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    int x = 0, y = 0, dir = 0, steps_taken = 0, segments_completed = 0, total = 0, curr_idx = 0;
    int steps_to_take = 1;
    int max_offsets = vr * vr;
    env->neighbor_offsets[curr_idx++] = 0;
    env->neighbor_offsets[curr_idx++] = 0;
    total++;
    // Generate spiral pattern
    while (total < max_offsets) {
        x += dx[dir];
        y += dy[dir];
        if (abs(x) <= vr / 2 && abs(y) <= vr / 2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        steps_taken++;
        if (steps_taken != steps_to_take) {
            continue;
        }
        steps_taken = 0;
        dir = (dir + 1) % 4; // Change direction (clockwise: right->up->left->down)
        segments_completed++;
        if (segments_completed % 2 == 0) {
            steps_to_take++;
        }
    }
}

static void cache_neighbor_offsets(Drive *env) {
    int count = 0;
    int cell_count = env->grid_map->grid_cols * env->grid_map->grid_rows;
    env->grid_map->neighbor_cache_entities = (GridMapEntity **) calloc(cell_count, sizeof(GridMapEntity *));
    env->grid_map->neighbor_cache_count = (int *) calloc(cell_count + 1, sizeof(int));
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols; // Convert to 2D coordinates
        int cell_y = i / env->grid_map->grid_cols;
        int current_cell_neighbor_count = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            current_cell_neighbor_count += grid_count;
        }
        env->grid_map->neighbor_cache_count[i] = current_cell_neighbor_count;
        count += current_cell_neighbor_count;
        if (current_cell_neighbor_count == 0) {
            env->grid_map->neighbor_cache_entities[i] = NULL;
            continue;
        }
        env->grid_map->neighbor_cache_entities[i]
            = (GridMapEntity *) calloc(current_cell_neighbor_count, sizeof(GridMapEntity));
    }

    env->grid_map->neighbor_cache_count[cell_count] = count;
    for (int i = 0; i < cell_count; i++) {
        int cell_x = i % env->grid_map->grid_cols;
        int cell_y = i / env->grid_map->grid_cols;
        int base_index = 0;
        for (int j = 0; j < env->grid_map->vision_range * env->grid_map->vision_range; j++) {
            int x = cell_x + env->neighbor_offsets[j * 2];
            int y = cell_y + env->neighbor_offsets[j * 2 + 1];
            int grid_index = env->grid_map->grid_cols * y + x;
            if (x < 0 || x >= env->grid_map->grid_cols || y < 0 || y >= env->grid_map->grid_rows) {
                continue;
            }
            int grid_count = env->grid_map->cell_entities_count[grid_index];
            // Skip if no entities or source is NULL
            if (grid_count == 0 || env->grid_map->cells[grid_index] == NULL) {
                continue;
            }
            // Copy grid_count pairs (entity_idx, geometry_idx) at once
            memcpy(
                &env->grid_map->neighbor_cache_entities[i][base_index],
                env->grid_map->cells[grid_index],
                grid_count * sizeof(GridMapEntity));
            base_index += grid_count;
        }
    }
}

static int get_neighbors_entities(
    Drive *env,
    float x,
    float y,
    GridMapEntity *entity_list,
    int max_size,
    const int (*local_offsets)[2],
    int offset_size) {
    int index = get_grid_index(env, x, y);
    if (index == -1) {
        return 0;
    }
    // Calculate 2D grid coordinates
    int cols = env->grid_map->grid_cols;
    int cell_x = index % cols;
    int cell_y = index / cols;
    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = cell_x + local_offsets[i][0];
        int ny = cell_y + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx < 0 || nx >= env->grid_map->grid_cols || ny < 0 || ny >= env->grid_map->grid_rows) {
            continue;
        }
        int neighbor_idx = ny * env->grid_map->grid_cols + nx;
        int count = env->grid_map->cell_entities_count[neighbor_idx];
        // Add entities from this cell to the list
        for (int j = 0; j < count && entity_list_count < max_size; j++) {
            entity_list[entity_list_count++] = env->grid_map->cells[neighbor_idx][j];
        }
    }
    return entity_list_count;
}

// ========================================
// Map Loading Functions
// ========================================

// Timing changes only during loading; temporal arrays are never in SharedMapData.
static int validate_replay_timing_config(Drive *drive) {
    if (!drive->resample_replay_to_dt) {
        return 0;
    }
    const char *error = NULL;
    if (drive->simulation_mode != SIMULATION_MODE_REPLAY) {
        error = "resample_replay_to_dt requires simulation_mode=replay";
    } else if (drive->init_step != 0 || drive->init_step_spread) {
        error = "resample_replay_to_dt requires init_step=0 and init_step_spread=false";
    } else if (!isfinite(drive->dt) || drive->dt <= 0) {
        error = "dt must be finite and positive";
    } else if (drive->scenario_length <= 0 || drive->scenario_length == INT_MAX) {
        error = "scenario_length must be positive and less than INT_MAX";
    }
    if (error != NULL) {
        snprintf(drive->load_error, sizeof(drive->load_error), "%s", error);
        return -1;
    }
    return 0;
}

static int prepare_replay_timing(Drive *drive) {
    if (!drive->resample_replay_to_dt) {
        return 0;
    }
    if (validate_replay_timing_config(drive) != 0) {
        return -1;
    }
    if (!isfinite(drive->log_dt) || drive->log_dt <= 0) {
        snprintf(drive->load_error, sizeof(drive->load_error), "log_dt must be finite and positive");
        return -1;
    }
    const double ratio_tolerance = 1e-5;
    double ratio = (double) drive->dt / drive->log_dt;
    double rounded_stride = round(ratio);
    if (!isfinite(ratio) || rounded_stride < 1 || rounded_stride > INT_MAX
        || fabs(ratio - rounded_stride) > ratio_tolerance) {
        snprintf(drive->load_error, sizeof(drive->load_error), "dt/log_dt must be a positive integer (got %.9g)", ratio);
        return -1;
    }
    int stride = (int) rounded_stride;
    int source_count = drive->log_length;
    if (source_count <= 0 || (source_count - 1) / stride < drive->scenario_length) {
        snprintf(drive->load_error, sizeof(drive->load_error),
                 "log_length=%d is insufficient for scenario_length=%d with stride=%d",
                 source_count, drive->scenario_length, stride);
        return -1;
    }
    for (int agent_idx = 0; agent_idx < drive->num_total_agents; agent_idx++) {
        if (drive->agents[agent_idx].trajectory_size != source_count) {
            snprintf(drive->load_error, sizeof(drive->load_error),
                     "agents[%d].trajectory_size=%d differs from log_length=%d", agent_idx,
                     drive->agents[agent_idx].trajectory_size, source_count);
            return -1;
        }
    }
    for (int traffic_idx = 0; traffic_idx < drive->num_traffic_elements; traffic_idx++) {
        int state_count = drive->traffic_elements[traffic_idx].state_size;
        if (state_count != 0 && state_count != source_count) {
            snprintf(drive->load_error, sizeof(drive->load_error),
                     "traffic_elements[%d].state_size=%d differs from log_length=%d",
                     traffic_idx, state_count, source_count);
            return -1;
        }
    }

    // Ascending compaction cannot overwrite a future source sample. Velocities
    // stay in m/s; sampling changes the time grid, never the physical units.
    int retained_count = (source_count - 1) / stride + 1;
    for (int agent_idx = 0; agent_idx < drive->num_total_agents; agent_idx++) {
        Agent *agent = &drive->agents[agent_idx];
        for (int sample_idx = 0; sample_idx < retained_count; sample_idx++) {
            int source_idx = sample_idx * stride;
            agent->log_trajectory_x[sample_idx] = agent->log_trajectory_x[source_idx];
            agent->log_trajectory_y[sample_idx] = agent->log_trajectory_y[source_idx];
            agent->log_trajectory_z[sample_idx] = agent->log_trajectory_z[source_idx];
            agent->log_heading[sample_idx] = agent->log_heading[source_idx];
            agent->log_velocity_x[sample_idx] = agent->log_velocity_x[source_idx];
            agent->log_velocity_y[sample_idx] = agent->log_velocity_y[source_idx];
            agent->log_length[sample_idx] = agent->log_length[source_idx];
            agent->log_width[sample_idx] = agent->log_width[source_idx];
            agent->log_height[sample_idx] = agent->log_height[source_idx];
            agent->log_valid[sample_idx] = agent->log_valid[source_idx];
        }
        agent->trajectory_size = retained_count;
    }
    for (int traffic_idx = 0; traffic_idx < drive->num_traffic_elements; traffic_idx++) {
        TrafficControlElement *traffic = &drive->traffic_elements[traffic_idx];
        if (traffic->state_size == 0) {
            continue;
        }
        for (int sample_idx = 0; sample_idx < retained_count; sample_idx++) {
            traffic->states[sample_idx] = traffic->states[sample_idx * stride];
        }
        traffic->state_size = retained_count;
    }
    drive->log_length = retained_count;
    drive->log_dt = drive->dt;
    return 0;
}

// Also used after a partial read: calloc leaves all unread pointers NULL.
static void free_loaded_map(Drive *drive) {
    for (int agent_idx = 0; drive->agents != NULL && agent_idx < drive->num_total_agents; agent_idx++) {
        free_agent(&drive->agents[agent_idx]);
    }
    for (int road_idx = 0; drive->road_elements != NULL && road_idx < drive->num_road_elements; road_idx++) {
        free_road_element(&drive->road_elements[road_idx]);
    }
    for (int traffic_idx = 0; drive->traffic_elements != NULL && traffic_idx < drive->num_traffic_elements; traffic_idx++) {
        free_traffic_element(&drive->traffic_elements[traffic_idx]);
    }
    free(drive->agents);
    free(drive->road_elements);
    free(drive->traffic_elements);
    free_lane_graph(&drive->lane_graph);
    memset(&drive->lane_graph, 0, sizeof(drive->lane_graph));
    free(drive->objects_of_interest);
    free(drive->tracks_to_predict);
    drive->agents = NULL;
    drive->road_elements = NULL;
    drive->traffic_elements = NULL;
    drive->objects_of_interest = NULL;
    drive->tracks_to_predict = NULL;
    drive->num_total_agents = 0;
    drive->num_road_elements = 0;
    drive->num_traffic_elements = 0;
}

// Binary counts are bounded by bytes remaining before allocation or bulk read.
static void *allocate_map_field(size_t count, size_t item_bytes, FILE *file, long file_bytes,
                                Drive *drive, const char *field, bool records) {
    long position = ftell(file);
    size_t disk_item_bytes = records ? sizeof(int) : item_bytes;
    if (position < 0 || position > file_bytes || count > SIZE_MAX / item_bytes
        || count > (size_t) (file_bytes - position) / disk_item_bytes) {
        snprintf(drive->load_error, sizeof(drive->load_error), "%s count exceeds remaining file bytes", field);
        return NULL;
    }
    // Empty optional arrays retain their zero length and own no allocation.
    if (count == 0) {
        return NULL;
    }
    void *allocation = records ? calloc(count, item_bytes) : malloc(count * item_bytes);
    if (allocation == NULL) {
        snprintf(drive->load_error, sizeof(drive->load_error), "%s allocation failed", field);
    }
    return allocation;
}

static size_t read_map_field(void *target, size_t item_bytes, size_t count, FILE *file,
                            long file_bytes, Drive *drive, const char *field) {
    long position = ftell(file);
    snprintf(drive->load_error, sizeof(drive->load_error), "%s invalid or truncated", field);
    if (target == NULL || position < 0 || position > file_bytes
        || count > (size_t) (file_bytes - position) / item_bytes) {
        return 0;
    }
    return fread(target, item_bytes, count, file);
}

int load_map_binary(const char *filename, Drive *drive) {
    if (validate_replay_timing_config(drive) != 0) {
        return -1;
    }
    snprintf(drive->load_error, sizeof(drive->load_error), "cannot open scenario");
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return -1;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        goto load_failure;
    }
    long file_bytes = ftell(file);
    if (file_bytes < 0 || fseek(file, 0, SEEK_SET) != 0) {
        goto load_failure;
    }
    int num_total_agents, num_roads, num_traffic, num_objects;
    if (read_map_field(
        &num_total_agents, sizeof(int), 1, file, file_bytes, drive, "&num_total_agents") != 1) {
        goto load_failure;
    }
    if (read_map_field(
        &num_roads, sizeof(int), 1, file, file_bytes, drive, "&num_roads") != 1) {
        goto load_failure;
    }
    if (read_map_field(
        &num_traffic, sizeof(int), 1, file, file_bytes, drive, "&num_traffic") != 1) {
        goto load_failure;
    }
    if (read_map_field(
        &num_objects, sizeof(int), 1, file, file_bytes, drive, "&num_objects") != 1) {
        goto load_failure;
    }

    if (num_total_agents < 0 || num_roads < 0 || num_traffic < 0 || num_objects < 0) {
        snprintf(drive->load_error, sizeof(drive->load_error), "negative header count");
        goto load_failure;
    }
    drive->num_total_agents = num_total_agents;
    drive->num_road_elements = num_roads;
    drive->num_traffic_elements = num_traffic;
    drive->num_objects = num_objects;

    if (num_total_agents > 0) {
        drive->agents = (Agent *) allocate_map_field(
            num_total_agents, sizeof(Agent), file, file_bytes, drive, "drive->agents", true);
        if (drive->agents == NULL && num_total_agents != 0) {
            goto load_failure;
        }
    }
    if (num_roads > 0) {
        drive->road_elements = (RoadMapElement *) allocate_map_field(
            num_roads, sizeof(RoadMapElement), file, file_bytes, drive, "drive->road_elements", true);
        if (drive->road_elements == NULL && num_roads != 0) {
            goto load_failure;
        }
    }
    if (num_traffic > 0) {
        drive->traffic_elements = (TrafficControlElement *) allocate_map_field(
            num_traffic, sizeof(TrafficControlElement), file, file_bytes, drive, "drive->traffic_elements", true);
        if (drive->traffic_elements == NULL && num_traffic != 0) {
            goto load_failure;
        }
    }

    for (int i = 0; i < num_total_agents; i++) {
        Agent *agent = &drive->agents[i];

        if (read_map_field(
            &agent->id, sizeof(int), 1, file, file_bytes, drive, "&agent->id") != 1) {
            goto load_failure;
        }
        if (agent->id != i) {
            printf("[ERROR] -> Agent id %d != idx %d. Binary must be reindexed (id == idx).\n", agent->id, i);
            goto load_failure;
        }
        if (read_map_field(
            &agent->type, sizeof(int), 1, file, file_bytes, drive, "&agent->type") != 1) {
            goto load_failure;
        }
        if (read_map_field(
            &agent->trajectory_size, sizeof(int), 1, file, file_bytes, drive, "&agent->trajectory_size") != 1) {
            goto load_failure;
        }

        int tlen = agent->trajectory_size;
        agent->log_trajectory_x = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_trajectory_x", false);
        if (agent->log_trajectory_x == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_trajectory_y = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_trajectory_y", false);
        if (agent->log_trajectory_y == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_trajectory_z = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_trajectory_z", false);
        if (agent->log_trajectory_z == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_heading = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_heading", false);
        if (agent->log_heading == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_velocity_x = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_velocity_x", false);
        if (agent->log_velocity_x == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_velocity_y = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_velocity_y", false);
        if (agent->log_velocity_y == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_length = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_length", false);
        if (agent->log_length == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_width = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_width", false);
        if (agent->log_width == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_height = (float *) allocate_map_field(
            tlen, sizeof(float), file, file_bytes, drive, "agent->log_height", false);
        if (agent->log_height == NULL && tlen != 0) {
            goto load_failure;
        }
        agent->log_valid = (int *) allocate_map_field(
            tlen, sizeof(int), file, file_bytes, drive, "agent->log_valid", false);
        if (agent->log_valid == NULL && tlen != 0) {
            goto load_failure;
        }

        if ((size_t) tlen > 0 && read_map_field(
            agent->log_trajectory_x, sizeof(float), tlen, file, file_bytes, drive, "agent->log_trajectory_x") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_trajectory_y, sizeof(float), tlen, file, file_bytes, drive, "agent->log_trajectory_y") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_trajectory_z, sizeof(float), tlen, file, file_bytes, drive, "agent->log_trajectory_z") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_heading, sizeof(float), tlen, file, file_bytes, drive, "agent->log_heading") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_velocity_x, sizeof(float), tlen, file, file_bytes, drive, "agent->log_velocity_x") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_velocity_y, sizeof(float), tlen, file, file_bytes, drive, "agent->log_velocity_y") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_length, sizeof(float), tlen, file, file_bytes, drive, "agent->log_length") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_width, sizeof(float), tlen, file, file_bytes, drive, "agent->log_width") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_height, sizeof(float), tlen, file, file_bytes, drive, "agent->log_height") != (size_t) tlen) {
            goto load_failure;
        }
        if ((size_t) tlen > 0 && read_map_field(
            agent->log_valid, sizeof(int), tlen, file, file_bytes, drive, "agent->log_valid") != (size_t) tlen) {
            goto load_failure;
        }

        if (read_map_field(
            &agent->route_length, sizeof(int), 1, file, file_bytes, drive, "&agent->route_length") != 1) {
            goto load_failure;
        }

        if (agent->route_length < 0) {
            snprintf(drive->load_error, sizeof(drive->load_error), "agent->route_length must be nonnegative");
            goto load_failure;
        }
        if (agent->route_length > 0) {
            agent->route = (int *) allocate_map_field(
                agent->route_length, sizeof(int), file, file_bytes, drive, "agent->route", false);
            if (agent->route == NULL && agent->route_length != 0) {
                goto load_failure;
            }
            if (read_map_field(
                agent->route, sizeof(int), agent->route_length, file, file_bytes, drive, "agent->route") != (size_t) agent->route_length) {
                goto load_failure;
            }
        } else {
            agent->route = NULL;
        }

        if (read_map_field(
            &agent->route_gt_len, sizeof(int), 1, file, file_bytes, drive, "&agent->route_gt_len") != 1) {
            goto load_failure;
        }

        if (read_map_field(
            &agent->gt_goal_x, sizeof(float), 1, file, file_bytes, drive, "&agent->gt_goal_x") != 1) {
            goto load_failure;
        }
        if (read_map_field(
            &agent->gt_goal_y, sizeof(float), 1, file, file_bytes, drive, "&agent->gt_goal_y") != 1) {
            goto load_failure;
        }
        if (read_map_field(
            &agent->gt_goal_z, sizeof(float), 1, file, file_bytes, drive, "&agent->gt_goal_z") != 1) {
            goto load_failure;
        }
        if (read_map_field(
            &agent->mark_as_expert, sizeof(int), 1, file, file_bytes, drive, "&agent->mark_as_expert") != 1) {
            goto load_failure;
        }
    }

    for (int i = 0; i < num_roads; i++) {
        RoadMapElement *road = &drive->road_elements[i];
        int road_id;

        if (read_map_field(
            &road_id, sizeof(int), 1, file, file_bytes, drive, "&road_id") != 1) {
            goto load_failure;
        }
        if (road_id != i) {
            printf("[ERROR] -> Road element id %d != idx %d. Binary must be reindexed (id == idx).\n", road_id, i);
            goto load_failure;
        }
        if (read_map_field(
            &road->type, sizeof(int), 1, file, file_bytes, drive, "&road->type") != 1) {
            goto load_failure;
        }
        if (read_map_field(
            &road->segment_size, sizeof(int), 1, file, file_bytes, drive, "&road->segment_size") != 1) {
            goto load_failure;
        }

        int slen = road->segment_size;

        road->x = (float *) allocate_map_field(
            slen, sizeof(float), file, file_bytes, drive, "road->x", false);
        if (road->x == NULL && slen != 0) {
            goto load_failure;
        }
        road->y = (float *) allocate_map_field(
            slen, sizeof(float), file, file_bytes, drive, "road->y", false);
        if (road->y == NULL && slen != 0) {
            goto load_failure;
        }
        road->z = (float *) allocate_map_field(
            slen, sizeof(float), file, file_bytes, drive, "road->z", false);
        if (road->z == NULL && slen != 0) {
            goto load_failure;
        }

        if ((size_t) slen > 0 && read_map_field(
            road->x, sizeof(float), slen, file, file_bytes, drive, "road->x") != (size_t) slen) {
            goto load_failure;
        }
        if ((size_t) slen > 0 && read_map_field(
            road->y, sizeof(float), slen, file, file_bytes, drive, "road->y") != (size_t) slen) {
            goto load_failure;
        }
        if ((size_t) slen > 0 && read_map_field(
            road->z, sizeof(float), slen, file, file_bytes, drive, "road->z") != (size_t) slen) {
            goto load_failure;
        }

        road->headings = (float *) allocate_map_field(
            slen, sizeof(float), file, file_bytes, drive, "road->headings", false);
        if (road->headings == NULL && slen != 0) {
            goto load_failure;
        }
        if ((size_t) slen > 0 && read_map_field(
            road->headings, sizeof(float), slen, file, file_bytes, drive, "road->headings") != (size_t) slen) {
            goto load_failure;
        }

        if (is_road_lane(road->type)) {
            if (read_map_field(
                &road->num_entries, sizeof(int), 1, file, file_bytes, drive, "&road->num_entries") != 1) {
                goto load_failure;
            }
            if (road->num_entries < 0) {
                snprintf(drive->load_error, sizeof(drive->load_error), "road->num_entries must be nonnegative");
                goto load_failure;
            }
            if (road->num_entries > 0) {
                road->entry_lanes = (int *) allocate_map_field(
                    road->num_entries, sizeof(int), file, file_bytes, drive, "road->entry_lanes", false);
                if (road->entry_lanes == NULL && road->num_entries != 0) {
                    goto load_failure;
                }
                if (read_map_field(
                    road->entry_lanes, sizeof(int), road->num_entries, file, file_bytes, drive, "road->entry_lanes") != (size_t) road->num_entries) {
                    goto load_failure;
                }
            } else {
                road->entry_lanes = NULL;
            }

            if (read_map_field(
                &road->num_exits, sizeof(int), 1, file, file_bytes, drive, "&road->num_exits") != 1) {
                goto load_failure;
            }
            if (road->num_exits < 0) {
                snprintf(drive->load_error, sizeof(drive->load_error), "road->num_exits must be nonnegative");
                goto load_failure;
            }
            if (road->num_exits > 0) {
                road->exit_lanes = (int *) allocate_map_field(
                    road->num_exits, sizeof(int), file, file_bytes, drive, "road->exit_lanes", false);
                if (road->exit_lanes == NULL && road->num_exits != 0) {
                    goto load_failure;
                }
                if (read_map_field(
                    road->exit_lanes, sizeof(int), road->num_exits, file, file_bytes, drive, "road->exit_lanes") != (size_t) road->num_exits) {
                    goto load_failure;
                }
            } else {
                road->exit_lanes = NULL;
            }

            if (read_map_field(
                &road->speed_limit, sizeof(float), 1, file, file_bytes, drive, "&road->speed_limit") != 1) {
                goto load_failure;
            }
            if (read_map_field(
                &road->length, sizeof(float), 1, file, file_bytes, drive, "&road->length") != 1) {
                goto load_failure;
            }
            road->cum_lengths = (float *) allocate_map_field(
                slen, sizeof(float), file, file_bytes, drive, "road->cum_lengths", false);
            if (road->cum_lengths == NULL && slen != 0) {
                goto load_failure;
            }
            if ((size_t) slen > 0 && read_map_field(
                road->cum_lengths, sizeof(float), slen, file, file_bytes, drive, "road->cum_lengths") != (size_t) slen) {
                goto load_failure;
            }
        } else {
            road->num_entries = 0;
            road->num_exits = 0;
            road->entry_lanes = NULL;
            road->exit_lanes = NULL;
            road->speed_limit = 0.0f;
            road->length = 0.0f;
            road->cum_lengths = NULL;
        }
    }

    for (int i = 0; i < num_traffic; i++) {
        TrafficControlElement *tc = &drive->traffic_elements[i];
        int traffic_id;

        if (read_map_field(
            &traffic_id, sizeof(int), 1, file, file_bytes, drive, "&traffic_id") != 1) {
            goto load_failure;
        }
        if (traffic_id != i) {
            printf(
                "[ERROR] -> Traffic element id %d != idx %d. Binary must be reindexed (id == idx).\n",
                traffic_id,
                i);
            goto load_failure;
        }
        if (read_map_field(
            &tc->type, sizeof(int), 1, file, file_bytes, drive, "&tc->type") != 1) {
            goto load_failure;
        }
        if (read_map_field(
            tc->stop_line, sizeof(float), 6, file, file_bytes, drive, "tc->stop_line") != 6) {
            goto load_failure;
        }
        if (read_map_field(
            &tc->heading, sizeof(float), 1, file, file_bytes, drive, "&tc->heading") != 1) {
            goto load_failure;
        }
        if (read_map_field(
            &tc->state_size, sizeof(int), 1, file, file_bytes, drive, "&tc->state_size") != 1) {
            goto load_failure;
        }

        int state_len = tc->state_size;

        tc->states = (int *) allocate_map_field(
            state_len, sizeof(int), file, file_bytes, drive, "tc->states", false);
        if (tc->states == NULL && state_len != 0) {
            goto load_failure;
        }
        if ((size_t) state_len > 0 && read_map_field(
            tc->states, sizeof(int), state_len, file, file_bytes, drive, "tc->states") != (size_t) state_len) {
            goto load_failure;
        }

        if (read_map_field(
            &tc->num_controlled_lanes, sizeof(int), 1, file, file_bytes, drive, "&tc->num_controlled_lanes") != 1) {
            goto load_failure;
        }
        if (tc->num_controlled_lanes < 0) {
            snprintf(drive->load_error, sizeof(drive->load_error), "tc->num_controlled_lanes must be nonnegative");
            goto load_failure;
        }
        if (tc->num_controlled_lanes > 0) {
            tc->controlled_lanes = (int *) allocate_map_field(
                tc->num_controlled_lanes, sizeof(int), file, file_bytes, drive, "tc->controlled_lanes", false);
            if (tc->controlled_lanes == NULL && tc->num_controlled_lanes != 0) {
                goto load_failure;
            }
            if (read_map_field(tc->controlled_lanes, sizeof(int), tc->num_controlled_lanes, file, file_bytes, drive, "tc->controlled_lanes")
                != (size_t) tc->num_controlled_lanes) {
                goto load_failure;
            }
        } else {
            tc->controlled_lanes = NULL;
        }
    }

    // Skip objects section
    int num_objects_features = 9; // x,y,z,heading,vx,vy,length,width,height (9 float arrays) + valid (1 int array)
    for (int i = 0; i < num_objects; i++) {
        int obj_id, obj_type, traj_len;
        if (read_map_field(&obj_id, sizeof(int), 1, file, file_bytes, drive, "&obj_id") != 1 || read_map_field(&obj_type, sizeof(int), 1, file, file_bytes, drive, "&obj_type") != 1
            || read_map_field(&traj_len, sizeof(int), 1, file, file_bytes, drive, "&traj_len") != 1) {
            goto load_failure;
        }
        // Skip: x,y,z,heading,vx,vy,length,width,height (9 float arrays) + valid (1 int array)
        long position = ftell(file);
        size_t object_sample_bytes = num_objects_features * sizeof(float) + sizeof(int);
        if (traj_len < 0 || position < 0 || (size_t) traj_len > (size_t) (file_bytes - position) / object_sample_bytes
            || fseek(file, (long) ((size_t) traj_len * object_sample_bytes), SEEK_CUR) != 0) {
            snprintf(drive->load_error, sizeof(drive->load_error), "objects trajectory length exceeds file");
            goto load_failure;
        }
    }

    // Lane graph section
    int n_lanes_graph;
    if (read_map_field(
        &n_lanes_graph, sizeof(int), 1, file, file_bytes, drive, "&n_lanes_graph") != 1) {
        goto load_failure;
    }
    drive->lane_graph.n_lanes = n_lanes_graph;
    drive->lane_graph.lane_ids = NULL;
    drive->lane_graph.distances = NULL;
    drive->lane_graph.lane_to_graph_idx = NULL;
    if (n_lanes_graph < 0) {
        snprintf(drive->load_error, sizeof(drive->load_error), "n_lanes_graph must be nonnegative");
        goto load_failure;
    }
    if (n_lanes_graph > 0) {
        drive->lane_graph.lane_ids = (int *) allocate_map_field(
            n_lanes_graph, sizeof(int), file, file_bytes, drive, "drive->lane_graph.lane_ids", false);
        if (drive->lane_graph.lane_ids == NULL && n_lanes_graph != 0) {
            goto load_failure;
        }
        if (read_map_field(
            drive->lane_graph.lane_ids, sizeof(int), n_lanes_graph, file, file_bytes, drive, "drive->lane_graph.lane_ids") != (size_t) n_lanes_graph) {
            goto load_failure;
        }
        drive->lane_graph.distances = (float *) allocate_map_field(
            (size_t) n_lanes_graph * n_lanes_graph, sizeof(float), file, file_bytes, drive, "drive->lane_graph.distances", false);
        if (drive->lane_graph.distances == NULL && (size_t) n_lanes_graph * n_lanes_graph != 0) {
            goto load_failure;
        }
        if (read_map_field(drive->lane_graph.distances, sizeof(float), (size_t) n_lanes_graph * n_lanes_graph, file, file_bytes, drive, "drive->lane_graph.distances")
            != (size_t) ((size_t) n_lanes_graph * n_lanes_graph)) {
            goto load_failure;
        }

        // Build reverse lookup road-element idx -> graph idx
        int num_roads = drive->num_road_elements;
        drive->lane_graph.lane_to_graph_idx = (int *) malloc((size_t) num_roads * sizeof(int));
        if (drive->lane_graph.lane_to_graph_idx == NULL && num_roads != 0) {
            goto load_failure;
        }
        for (int r = 0; r < num_roads; r++) {
            drive->lane_graph.lane_to_graph_idx[r] = -1;
        }
        for (int g = 0; g < n_lanes_graph; g++) {
            int lane_idx = drive->lane_graph.lane_ids[g];
            if (lane_idx < 0 || lane_idx >= num_roads) {
                printf("[ERROR] -> lane_graph lane_id %d out of range [0,%d).\n", lane_idx, num_roads);
                goto load_failure;
            }
            drive->lane_graph.lane_to_graph_idx[lane_idx] = g;
        }
    }

    // Metadata
    if (read_map_field(
        drive->scenario_id, sizeof(char), 128, file, file_bytes, drive, "drive->scenario_id") != 128) {
        goto load_failure;
    }
    if (read_map_field(
        drive->dataset_name, sizeof(char), 32, file, file_bytes, drive, "drive->dataset_name") != 32) {
        goto load_failure;
    }
    if (read_map_field(
        &drive->log_length, sizeof(int), 1, file, file_bytes, drive, "&drive->log_length") != 1) {
        goto load_failure;
    }
    if (read_map_field(
        &drive->log_dt, sizeof(float), 1, file, file_bytes, drive, "&drive->log_dt") != 1) {
        goto load_failure;
    }
    if (read_map_field(
        &drive->num_objects_of_interest, sizeof(int), 1, file, file_bytes, drive, "&drive->num_objects_of_interest") != 1) {
        goto load_failure;
    }

    if (drive->num_objects_of_interest < 0) {
        snprintf(drive->load_error, sizeof(drive->load_error), "drive->num_objects_of_interest must be nonnegative");
        goto load_failure;
    }
    if (drive->num_objects_of_interest > 0) {
        drive->objects_of_interest = (int *) allocate_map_field(
            drive->num_objects_of_interest, sizeof(int), file, file_bytes, drive, "drive->objects_of_interest", false);
        if (drive->objects_of_interest == NULL && drive->num_objects_of_interest != 0) {
            goto load_failure;
        }
        if (read_map_field(drive->objects_of_interest, sizeof(int), drive->num_objects_of_interest, file, file_bytes, drive, "drive->objects_of_interest")
            != (size_t) drive->num_objects_of_interest) {
            goto load_failure;
        }
    } else {
        drive->objects_of_interest = NULL;
    }

    if (read_map_field(
        &drive->num_tracks_to_predict, sizeof(int), 1, file, file_bytes, drive, "&drive->num_tracks_to_predict") != 1) {
        goto load_failure;
    }

    if (drive->num_tracks_to_predict < 0) {
        snprintf(drive->load_error, sizeof(drive->load_error), "drive->num_tracks_to_predict must be nonnegative");
        goto load_failure;
    }
    if (drive->num_tracks_to_predict > 0) {
        drive->tracks_to_predict = (int *) allocate_map_field(
            drive->num_tracks_to_predict, sizeof(int), file, file_bytes, drive, "drive->tracks_to_predict", false);
        if (drive->tracks_to_predict == NULL && drive->num_tracks_to_predict != 0) {
            goto load_failure;
        }
        if (read_map_field(drive->tracks_to_predict, sizeof(int), drive->num_tracks_to_predict, file, file_bytes, drive, "drive->tracks_to_predict")
            != (size_t) drive->num_tracks_to_predict) {
            goto load_failure;
        }
    } else {
        drive->tracks_to_predict = NULL;
    }

    if (drive->resample_replay_to_dt && ftell(file) != file_bytes) {
        snprintf(drive->load_error, sizeof(drive->load_error), "trailing bytes after tracks_to_predict");
        goto load_failure;
    }
    if (prepare_replay_timing(drive) != 0) {
        goto load_failure;
    }
    fclose(file);
    drive->load_error[0] = '\0';
    return 0;

load_failure:
    fclose(file);
    free_loaded_map(drive);
    return -1;
}


static struct SharedMapData *map_cache_lookup(Drive *env) {
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] != NULL && strcmp(g_map_cache[i]->map_name, env->map_name) == 0
            && g_map_cache[i]->obs_lane_stride == env->obs_lane_stride
            && g_map_cache[i]->obs_boundary_stride == env->obs_boundary_stride) {
            return g_map_cache[i];
        }
    }
    return NULL;
}

static void map_cache_insert(struct SharedMapData *entry) {
    // Reuse a NULL slot left by free_shared_map_data before growing the array.
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] == NULL) {
            g_map_cache[i] = entry;
            return;
        }
    }
    g_map_cache
        = (struct SharedMapData **) realloc(g_map_cache, (g_map_cache_count + 1) * sizeof(struct SharedMapData *));
    g_map_cache[g_map_cache_count++] = entry;
}

static void free_grid_map(GridMap *grid_map) {
    if (grid_map == NULL) {
        return;
    }
    int grid_cell_count = grid_map->grid_cols * grid_map->grid_rows;
    for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
        free(grid_map->cells[grid_index]);
    }
    free(grid_map->cells);
    free(grid_map->cell_entities_count);
    free(grid_map->grid_index_drivable);
    if (grid_map->neighbor_cache_entities != NULL) {
        for (int grid_index = 0; grid_index < grid_cell_count; grid_index++) {
            free(grid_map->neighbor_cache_entities[grid_index]);
        }
    }
    free(grid_map->neighbor_cache_entities);
    free(grid_map->neighbor_cache_count);
    free(grid_map);
}

// Free a shared geometry entry and everything it owns, and clear its cache slot.
// Only call in the building process.
static void free_shared_map_data(struct SharedMapData *shared) {
    if (shared == NULL) {
        return;
    }
    for (int i = 0; i < shared->num_road_elements; i++) {
        free_road_element(&shared->road_elements[i]);
    }
    free(shared->road_elements);
    free_grid_map(shared->grid_map);
    free(shared->neighbor_offsets);
    free_lane_graph(&shared->lane_graph);
    free(shared->map_name);
    for (int i = 0; i < g_map_cache_count; i++) {
        if (g_map_cache[i] == shared) {
            g_map_cache[i] = NULL;
            break;
        }
    }
    free(shared);
}

#endif
