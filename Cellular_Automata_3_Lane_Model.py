import numpy as np
import matplotlib.pyplot as plt
import random

def init_cars_array_efficient(density, road_length, lanes, max_speed, car_types):
    occupancy = np.zeros((lanes, road_length), dtype=bool)
    total_cells = road_length * lanes
    target_occupied = int(density * total_cells)
    lane_list = []
    pos_list = []
    speed_list = []
    size_list = []
    def valid_positions_for_lane(lane, car_size):
        valid = []
        for i in range(road_length):
            is_valid = True
            for j in range(car_size):
                if occupancy[lane, (i - j) % road_length]:
                    is_valid = False
                    break
            if is_valid:
                valid.append(i)
        return valid
    while occupancy.sum() < target_occupied:
        candidates = []
        for lane in range(lanes):
            for (ctype, car_size, weight) in car_types:
                valid = valid_positions_for_lane(lane, car_size)
                for pos in valid:
                    candidates.append((lane, pos, car_size))
        if not candidates:
            break
        lane_candidate, pos_candidate, car_size_candidate = random.choice(candidates)
        lane_list.append(lane_candidate)
        pos_list.append(pos_candidate)
        speed_list.append(random.randint(0, max_speed))
        size_list.append(car_size_candidate)
        for j in range(car_size_candidate):
            occupancy[lane_candidate, (pos_candidate - j) % road_length] = True
    return lane_list, pos_list, speed_list, size_list

def build_grid(lane_arr, pos_arr, size_arr, lanes, road_length):
    grid = -np.ones((lanes, road_length), dtype=int)
    n = len(pos_arr)
    for i in range(n):
        L = lane_arr[i]
        for j in range(size_arr[i]):
            cell = (pos_arr[i] - j) % road_length
            grid[L, cell] = i
    return grid

def get_gap_ahead(lane, pos, grid, road_length):
    gap = 0
    for d in range(1, road_length):
        cell = (pos + d) % road_length
        if grid[lane, cell] == -1:
            gap += 1
        else:
            break
    return gap

def get_gap_behind(lane, pos, grid, road_length):
    gap = 0
    for d in range(1, road_length):
        cell = (pos - d) % road_length
        if grid[lane, cell] == -1:
            gap += 1
        else:
            break
    return gap

def simulation_step(lane_arr, pos_arr, speed_arr, size_arr, road_length, lanes, max_speed, p_slow, threshold_gap, safe_gap_front, safe_gap_rear):
    n = len(pos_arr)
    grid = build_grid(lane_arr, pos_arr, size_arr, lanes, road_length)
    for i in range(n):
        if speed_arr[i] >= max_speed:
            continue
        current_lane = lane_arr[i]
        current_pos = pos_arr[i]
        current_gap = get_gap_ahead(current_lane, current_pos, grid, road_length)
        candidate_lane = None
        candidate_gap = -1
        for delta in [-1, 1]:
            new_lane = current_lane + delta
            if new_lane < 0 or new_lane >= lanes:
                continue
            block_free = True
            for j in range(size_arr[i]):
                if grid[new_lane, (current_pos - j) % road_length] != -1:
                    block_free = False
                    break
            if not block_free:
                continue
            gap_ahead_new = get_gap_ahead(new_lane, current_pos, grid, road_length)
            new_tail = (current_pos - size_arr[i] + 1) % road_length
            gap_behind_new = get_gap_behind(new_lane, new_tail, grid, road_length)
            if (gap_ahead_new >= current_gap + threshold_gap and gap_ahead_new >= safe_gap_front and gap_behind_new >= safe_gap_rear):
                if gap_ahead_new > candidate_gap:
                    candidate_lane = new_lane
                    candidate_gap = gap_ahead_new
        if candidate_lane is not None:
            lane_arr[i] = candidate_lane
    grid = build_grid(lane_arr, pos_arr, size_arr, lanes, road_length)
    for i in range(n):
        speed_arr[i] = min(speed_arr[i] + 1, max_speed)
        gap = get_gap_ahead(lane_arr[i], pos_arr[i], grid, road_length)
        if speed_arr[i] > gap:
            speed_arr[i] = gap
        if speed_arr[i] > 1 and random.random() < p_slow:
            speed_arr[i] = max(speed_arr[i] - 2, 0)
    for i in range(n):
        pos_arr[i] = (pos_arr[i] + speed_arr[i]) % road_length

def simulate_complex(density, road_length, lanes, max_speed, p_slow, steps, transient, threshold_gap, safe_gap_front, safe_gap_rear, car_types):
    lane_list, pos_list, speed_list, size_list = init_cars_array_efficient(density, road_length, lanes, max_speed, car_types)
    lane_arr = np.array(lane_list)
    pos_arr = np.array(pos_list)
    speed_arr = np.array(speed_list)
    size_arr = np.array(size_list)
    speed_records = []
    for step in range(steps):
        simulation_step(lane_arr, pos_arr, speed_arr, size_arr, road_length, lanes, max_speed, p_slow, threshold_gap, safe_gap_front, safe_gap_rear)
        if step >= transient:
            if len(speed_arr) > 0:
                speed_records.append(np.mean(speed_arr))
    avg_speed = np.mean(speed_records) if speed_records else 0
    flow = density * avg_speed
    return flow, avg_speed

def compute_flow_for_density_complex_array(road_length, lanes, density, max_speed, p_slow, steps, transient, threshold_gap, safe_gap_front, safe_gap_rear, car_types):
    return simulate_complex(density, road_length, lanes, max_speed, p_slow, steps, transient, threshold_gap, safe_gap_front, safe_gap_rear, car_types)

def plot_flow_vs_density_complex_array(road_length, lanes, max_speed, p_slow, steps, densities, transient, num_samples, threshold_gap, safe_gap_front, safe_gap_rear, car_types):
    all_densities = []
    all_flows = []
    mean_flows = []
    for d in densities:
        flows_for_d = []
        for _ in range(num_samples):
            flow, _ = compute_flow_for_density_complex_array(road_length, lanes, d, max_speed, p_slow, steps, transient, threshold_gap, safe_gap_front, safe_gap_rear, car_types)
            flows_for_d.append(flow)
            all_densities.append(d)
            all_flows.append(flow)
        mean_flow = np.mean(flows_for_d)
        mean_flows.append(mean_flow)
    plt.xlim(0, 0.9)
    plt.ylim(0, 1.2)
    plt.scatter(all_densities, all_flows, color='blue', alpha=0.6, label='Samples', s=2)
    plt.xlabel('Density (fraction of cells occupied)')
    plt.ylabel('Flow (vehicles per time step)')
    plt.title('Fundamental Diagram: Flow vs Density (Complex Model)')
    plt.grid(True)
    plt.legend()
    plt.show()
    mean_flows = np.array(mean_flows)
    max_index = np.argmax(mean_flows)
    max_mean_flow = mean_flows[max_index]
    max_flow_density = densities[max_index]
    return max_mean_flow, max_flow_density
# runs model and plots flow vs density (variables can be changed below)
if __name__ == "__main__":
    road_length = 100
    lanes = 3
    max_speed = 7
    p_slow = 0.2
    steps = 100
    transient = 50
    threshold_gap = 3
    safe_gap_front = 2
    safe_gap_rear = 2
    car_types = [('normal', 2, 0.7), ('van', 3, 0.2), ('lorry', 5, 0.1)]
    densities = np.linspace(0.09, 0.7, 100)
    max_flow, max_flow_density = plot_flow_vs_density_complex_array(road_length, lanes, max_speed, p_slow, steps, densities, transient, num_samples=10, threshold_gap=threshold_gap, safe_gap_front=safe_gap_front, safe_gap_rear=safe_gap_rear, car_types=car_types)
    print(f"Maximum Mean Flow is {max_flow:.2f} at Density {max_flow_density:.2f}")
