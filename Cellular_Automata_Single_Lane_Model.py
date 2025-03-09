import numpy as np
import matplotlib.pyplot as plt


def compute_gaps(positions, road_length):
    gaps = np.empty_like(positions)
    if len(positions) > 0:
        gaps[:-1] = positions[1:] - positions[:-1] - 1
        gaps[-1] = road_length - positions[-1] + positions[0] - 1
    return gaps


def simulate_ca(road_length, density, max_speed, p_slow, steps):
    num_cars = int(road_length * density)
    positions = np.sort(np.random.choice(road_length, num_cars, replace=False))
    gaps = compute_gaps(positions, road_length)
    speeds = np.minimum(max_speed, gaps).astype(int)

    space_time = []
    state = np.full(road_length, np.nan)
    state[positions] = speeds
    space_time.append(state.copy())

    for _ in range(steps):
        speeds = np.where(speeds < max_speed, speeds + 1, speeds)

        order = np.argsort(positions)
        positions = positions[order]
        speeds = speeds[order]

        gaps = compute_gaps(positions, road_length)
        speeds = np.minimum(speeds, gaps).astype(int)

        slowdown = (speeds > 0) & (np.random.rand(len(speeds)) < p_slow)
        speeds[slowdown] -= 1

        positions = (positions + speeds) % road_length

        order = np.argsort(positions)
        positions = positions[order]
        speeds = speeds[order]

        state = np.full(road_length, np.nan)
        state[positions] = speeds
        space_time.append(state.copy())

    return space_time


def plot_space_time(space_time, road_length, max_speed):
    fig, ax = plt.subplots(figsize=(8, 8))
    for t, state in enumerate(space_time):
        pos = np.where(~np.isnan(state))[0]
        if pos.size:
            spd = state[pos]
            ax.scatter(pos, np.full_like(pos, t), color=plt.cm.RdYlGn(spd / max_speed), s=5)
    ax.set_xlabel('Road Position')
    ax.set_ylabel('Time Step')
    ax.set_title('Space-Time Diagram of Traffic Flow (Closed Loop)')
    ax.invert_yaxis()
    ax.set_xlim(0, road_length)
    ax.set_ylim(len(space_time) - 1, 0)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def compute_flow_for_density(road_length, density, max_speed, p_slow, steps, transient=100):
    space_time = simulate_ca(road_length, density, max_speed, p_slow, steps)
    speeds_list = []
    for state in space_time[transient:]:
        speeds = state[~np.isnan(state)]
        if speeds.size > 0:
            speeds_list.append(np.mean(speeds))
    avg_speed = np.mean(speeds_list) if speeds_list else 0
    flow = density * avg_speed
    return flow, avg_speed


def plot_flow_vs_density(road_length, max_speed, p_slow, steps, densities, transient=100, num_samples=10):
    all_densities = []
    all_flows = []
    mean_flows = []

    for d in densities:
        flows_for_d = []
        for _ in range(num_samples):
            flow, _ = compute_flow_for_density(road_length, d, max_speed, p_slow, steps, transient)
            flows_for_d.append(flow)
            all_densities.append(d)
            all_flows.append(flow)
        mean_flows.append(np.mean(flows_for_d))

    plt.xlim(0, 0.9)
    plt.ylim(0, 1.2)
    plt.scatter(all_densities, all_flows, color='blue', label='Samples', alpha=0.6, s=1)

    plt.xlabel('Density (Fraction of Road Occupied)')
    plt.ylabel('Flow (Vehicles per Time Step)')
    plt.title('Fundamental Diagram: Flow vs Density')
    plt.grid(True)
    plt.legend()
    plt.show()

    mean_flows = np.array(mean_flows)
    max_index = np.argmax(mean_flows)
    max_mean_flow = mean_flows[max_index]
    max_flow_density = densities[max_index]

    return max_mean_flow, max_flow_density

# runs and plots the flow vs density
if __name__ == '__main__':
    road_length = 100
    density = 0.1
    max_speed = 9
    p_slow = 0.2
    steps = 100

    space_time = simulate_ca(road_length, density, max_speed, p_slow, steps)
    plot_space_time(space_time, road_length, max_speed)

    density_values = np.linspace(0.05, 0.7, 100)
    max_mean_flow, max_flow_density = plot_flow_vs_density(road_length, max_speed, p_slow, steps, density_values,
                                                           transient=50, num_samples=10)
    print(f"Maximum Mean Flow is {max_mean_flow:.3f} at Density {max_flow_density:.3f}")