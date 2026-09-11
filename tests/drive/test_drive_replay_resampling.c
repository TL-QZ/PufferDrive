#include "include/drive_fixture.h"
#include "include/test.h"

static int test_exact_samples(void) {
    Drive original = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    Drive sampled = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    sampled.resample_replay_to_dt = true;
    sampled.dt = 0.3f;
    sampled.scenario_length = 66;
    EXPECT_EQ_INT(load_map_binary(original.map_name, &original), 0);
    EXPECT_EQ_INT(load_map_binary(sampled.map_name, &sampled), 0);
    EXPECT_EQ_INT(original.log_length, 201);
    EXPECT_EQ_INT(sampled.log_length, 67);
    EXPECT_TRUE(sampled.log_dt == sampled.dt);
    for (int agent_idx = 0; agent_idx < original.num_total_agents; agent_idx++) {
        Agent *source = &original.agents[agent_idx];
        Agent *target = &sampled.agents[agent_idx];
        EXPECT_EQ_INT(target->trajectory_size, 67);
        EXPECT_EQ_INT(target->id, source->id);
        EXPECT_EQ_INT(target->route_length, source->route_length);
        EXPECT_TRUE(target->gt_goal_x == source->gt_goal_x);
        for (int sample_idx = 0; sample_idx < 67; sample_idx++) {
            int source_idx = sample_idx * 3;
            EXPECT_TRUE(target->log_trajectory_x[sample_idx] == source->log_trajectory_x[source_idx]);
            EXPECT_TRUE(target->log_trajectory_y[sample_idx] == source->log_trajectory_y[source_idx]);
            EXPECT_TRUE(target->log_trajectory_z[sample_idx] == source->log_trajectory_z[source_idx]);
            EXPECT_TRUE(target->log_heading[sample_idx] == source->log_heading[source_idx]);
            EXPECT_TRUE(target->log_velocity_x[sample_idx] == source->log_velocity_x[source_idx]);
            EXPECT_TRUE(target->log_velocity_y[sample_idx] == source->log_velocity_y[source_idx]);
            EXPECT_TRUE(target->log_length[sample_idx] == source->log_length[source_idx]);
            EXPECT_TRUE(target->log_width[sample_idx] == source->log_width[source_idx]);
            EXPECT_TRUE(target->log_height[sample_idx] == source->log_height[source_idx]);
            EXPECT_EQ_INT(target->log_valid[sample_idx], source->log_valid[source_idx]);
        }
    }
    for (int traffic_idx = 0; traffic_idx < original.num_traffic_elements; traffic_idx++) {
        TrafficControlElement *source = &original.traffic_elements[traffic_idx];
        TrafficControlElement *target = &sampled.traffic_elements[traffic_idx];
        EXPECT_EQ_INT(target->state_size, source->state_size == 0 ? 0 : 67);
        for (int sample_idx = 0; sample_idx < target->state_size; sample_idx++) {
            EXPECT_EQ_INT(target->states[sample_idx], source->states[sample_idx * 3]);
        }
    }
    c_close(&original);
    c_close(&sampled);
    return 0;
}

static int test_invalid_timing_and_lengths(void) {
    Drive env = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 0);
    EXPECT_EQ_INT(load_map_binary(env.map_name, &env), 0);
    env.resample_replay_to_dt = true;
    env.scenario_length = 66;
    env.dt = 0.25f;
    EXPECT_EQ_INT(prepare_replay_timing(&env), -1);
    EXPECT_TRUE(strstr(env.load_error, "dt/log_dt") != NULL);
    env.dt = 0.3f;
    env.log_dt = NAN;
    EXPECT_EQ_INT(prepare_replay_timing(&env), -1);
    env.log_dt = 0.1f;
    env.agents[0].trajectory_size--;
    EXPECT_EQ_INT(prepare_replay_timing(&env), -1);
    EXPECT_TRUE(strstr(env.load_error, "trajectory_size") != NULL);
    env.agents[0].trajectory_size++;
    EXPECT_TRUE(env.num_traffic_elements > 0);
    int old_size = env.traffic_elements[0].state_size;
    env.traffic_elements[0].state_size = 200;
    EXPECT_EQ_INT(prepare_replay_timing(&env), -1);
    EXPECT_TRUE(strstr(env.load_error, "state_size") != NULL);
    env.traffic_elements[0].state_size = old_size;
    env.scenario_length = 67;
    EXPECT_EQ_INT(prepare_replay_timing(&env), -1);
    EXPECT_TRUE(strstr(env.load_error, "insufficient") != NULL);
    c_close(&env);
    return 0;
}

static int test_cache_and_export_boundary(void) {
    Drive original = drive_test_make_env(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 1);
    Drive sampled = drive_test_env_config(drive_nuplan_map(), SIMULATION_MODE_REPLAY, 1, 1);
    sampled.resample_replay_to_dt = true;
    sampled.dt = 0.3f;
    sampled.scenario_length = 66;
    sampled.non_sdc_controller = CONTROLLER_REPLAY;
    allocate(&sampled);
    c_reset(&sampled);
    EXPECT_TRUE(sampled.shared_map == original.shared_map);
    EXPECT_TRUE(sampled.agents[0].log_trajectory_x != original.agents[0].log_trajectory_x);
    EXPECT_EQ_INT(original.agents[0].trajectory_size, 201);
    EXPECT_EQ_INT(sampled.agents[0].trajectory_size, 67);
    float x[68] = {0}, y[68] = {0}, z[68] = {0}, heading[68] = {0};
    int valid[68] = {0}, id[1], scenario_id[1];
    x[67] = y[67] = z[67] = heading[67] = 12345.0f;
    valid[67] = 12345;
    c_get_global_ground_truth_trajectories(&sampled, x, y, z, heading, valid, id, scenario_id);
    EXPECT_TRUE(x[67] == 12345.0f && y[67] == 12345.0f && z[67] == 12345.0f && heading[67] == 12345.0f);
    EXPECT_EQ_INT(valid[67], 12345);
    for (int step_idx = 0; step_idx < 132; step_idx++) {
        drive_set_neutral_actions(&sampled);
        c_step(&sampled);
    }
    EXPECT_EQ_INT(sampled.agents[0].trajectory_size, 67);
    EXPECT_EQ_INT(original.agents[0].trajectory_size, 201);
    free_allocated(&sampled);
    free_allocated(&original);
    EXPECT_EQ_INT(drive_map_cache_live_count(), 0);
    drive_map_cache_clear();
    return 0;
}

static int test_partial_load_cleanup(void) {
    FILE *source = fopen(drive_nuplan_map(), "rb");
    EXPECT_TRUE(source != NULL);
    EXPECT_EQ_INT(fseek(source, 0, SEEK_END), 0);
    long file_bytes = ftell(source);
    EXPECT_TRUE(file_bytes > 0);
    rewind(source);
    char *data = malloc(file_bytes);
    EXPECT_TRUE(data != NULL);
    EXPECT_TRUE(fread(data, 1, file_bytes, source) == (size_t) file_bytes);
    fclose(source);
    char filename[] = "/tmp/drive_resampling_XXXXXX";
    int descriptor = mkstemp(filename);
    EXPECT_TRUE(descriptor >= 0);
    close(descriptor);
    size_t truncated_sizes[] = {2, 1000, (size_t) file_bytes - 1};
    for (int case_idx = 0; case_idx < 3; case_idx++) {
        FILE *output = fopen(filename, "wb");
        EXPECT_TRUE(output != NULL);
        EXPECT_TRUE(fwrite(data, 1, truncated_sizes[case_idx], output) == truncated_sizes[case_idx]);
        fclose(output);
        Drive env = drive_test_env_config(filename, SIMULATION_MODE_REPLAY, 1, 0);
        env.resample_replay_to_dt = true;
        env.dt = 0.3f;
        env.scenario_length = 66;
        EXPECT_EQ_INT(load_map_binary(filename, &env), -1);
        EXPECT_TRUE(env.agents == NULL && env.road_elements == NULL && env.traffic_elements == NULL);
        EXPECT_EQ_INT(env.num_total_agents, 0);
        c_close(&env);
    }
    free(data);
    EXPECT_EQ_INT(unlink(filename), 0);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_exact_samples);
    RUN_TEST(test_invalid_timing_and_lengths);
    RUN_TEST(test_cache_and_export_boundary);
    RUN_TEST(test_partial_load_cleanup);
    return test_summary(failures);
}
