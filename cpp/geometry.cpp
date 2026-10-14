#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace py = pybind11;
using namespace pybind11::literals;

// Pairwise minimum distance between two trajectories
float pairwise_min_distance(
    const py::array_t<float>& traj1,
    const py::array_t<float>& traj2
) {
    auto buf1 = traj1.request();
    auto buf2 = traj2.request();

    float* ptr1 = static_cast<float*>(buf1.ptr);
    float* ptr2 = static_cast<float*>(buf2.ptr);

    int n1 = buf1.shape[0];
    int n2 = buf2.shape[0];

    float min_dist = std::numeric_limits<float>::max();

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            float dx = ptr1[i * 2] - ptr2[j * 2];
            float dy = ptr1[i * 2 + 1] - ptr2[j * 2 + 1];
            float dist = std::sqrt(dx * dx + dy * dy);
            min_dist = std::min(min_dist, dist);
        }
    }

    return min_dist;
}

// Time to collision between two trajectories
float time_to_collision(
    const py::array_t<float>& ego_traj,
    const py::array_t<float>& other_traj
) {
    auto buf_ego = ego_traj.request();
    auto buf_other = other_traj.request();

    float* ego_ptr = static_cast<float*>(buf_ego.ptr);
    float* other_ptr = static_cast<float*>(buf_other.ptr);

    int n_ego = buf_ego.shape[0];
    int n_other = buf_other.shape[0];
    int n_steps = std::min(n_ego, n_other);

    float min_ttc = 1000.0f;

    for (int i = 1; i < n_steps; i++) {
        float ego_x = ego_ptr[i * 2];
        float ego_y = ego_ptr[i * 2 + 1];
        float ego_vx = ego_ptr[i * 2] - ego_ptr[(i-1) * 2];
        float ego_vy = ego_ptr[i * 2 + 1] - ego_ptr[(i-1) * 2 + 1];

        float other_x = other_ptr[i * 2];
        float other_y = other_ptr[i * 2 + 1];
        float other_vx = other_ptr[i * 2] - other_ptr[(i-1) * 2];
        float other_vy = other_ptr[i * 2 + 1] - other_ptr[(i-1) * 2 + 1];

        float dx = other_x - ego_x;
        float dy = other_y - ego_y;
        float dvx = other_vx - ego_vx;
        float dvy = other_vy - ego_vy;

        float dist = std::sqrt(dx * dx + dy * dy);
        float closing_speed = -(dx * dvx + dy * dvy) / std::max(dist, 0.1f);

        if (closing_speed > 0.1f) {
            float ttc = dist / closing_speed;
            min_ttc = std::min(min_ttc, ttc);
        }
    }

    return min_ttc;
}

// Lateral offset from centerline
float lateral_offset_from_centerline(
    const py::array_t<float>& position,
    const py::array_t<float>& centerline
) {
    auto buf_pos = position.request();
    auto buf_line = centerline.request();

    float* pos_ptr = static_cast<float*>(buf_pos.ptr);
    float* line_ptr = static_cast<float*>(buf_line.ptr);

    float px = pos_ptr[0];
    float py = pos_ptr[1];

    int n_line = buf_line.shape[0];
    float min_offset = std::numeric_limits<float>::max();

    for (int i = 0; i < n_line - 1; i++) {
        float x1 = line_ptr[i * 2];
        float y1 = line_ptr[i * 2 + 1];
        float x2 = line_ptr[(i + 1) * 2];
        float y2 = line_ptr[(i + 1) * 2 + 1];

        float dx = x2 - x1;
        float dy = y2 - y1;
        float len_sq = dx * dx + dy * dy;

        if (len_sq < 0.001f) continue;

        float t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        t = std::max(0.0f, std::min(1.0f, t));

        float closest_x = x1 + t * dx;
        float closest_y = y1 + t * dy;

        float offset = std::sqrt(
            std::pow(px - closest_x, 2) + std::pow(py - closest_y, 2)
        );
        min_offset = std::min(min_offset, offset);
    }

    return min_offset;
}

// Heading delta in radians
float heading_delta(float h1, float h2) {
    float delta = h2 - h1;
    while (delta > M_PI) delta -= 2 * M_PI;
    while (delta < -M_PI) delta += 2 * M_PI;
    return std::abs(delta);
}

// Batch interaction features
py::list interaction_features(
    const py::array_t<float>& ego_traj,
    const py::array_t<float>& other_traj
) {
    py::dict result;
    result["min_gap"] = pairwise_min_distance(ego_traj, other_traj);
    result["min_ttc"] = time_to_collision(ego_traj, other_traj);
    result["closing_rate"] = 0.5f;  // Placeholder
    result["lateral_encroachment"] = 0.0f;  // Placeholder

    py::list ret;
    ret.append(result);
    return ret;
}

PYBIND11_MODULE(skg_geom, m) {
    m.def("pairwise_min_distance", &pairwise_min_distance, "Minimum distance between trajectories");
    m.def("time_to_collision", &time_to_collision, "Time to collision");
    m.def("lateral_offset_from_centerline", &lateral_offset_from_centerline, "Lateral offset from lane centerline");
    m.def("heading_delta", &heading_delta, "Heading difference");
    m.def("interaction_features", &interaction_features, "Batch interaction features");
}
