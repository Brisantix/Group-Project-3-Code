import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import random


class Car:
    def __init__(self, car_id, car_type, size, lane, position, speed, color):
        self.id = car_id
        self.car_type = car_type
        self.size = size
        self.lane = lane
        self.position = position
        self.speed = speed
        self.color = color

    def occupied_cells(self, road_length):
        cells = [(self.position - i) % road_length for i in range(self.size)]
        return sorted(cells)


class TrafficSimulation:
    def __init__(self, road_length=200, density=0.1, max_speed=5, p_slow=0.3, steps=200):
        self.road_length = road_length
        self.density = density
        self.max_speed = max_speed
        self.p_slow = p_slow
        self.steps = steps
        self.lanes = 3
        self.cars = []
        self.car_counter = 0

        self.threshold_gap = 3
        self.safe_gap_front = 2
        self.safe_gap_rear = 2

        self.car_types = [('normal', 2, 0.7), ('van', 3, 0.2), ('lorry', 5, 0.1)]
        self.possible_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

        self._initialize_cars()

    def _initialize_cars(self):
        total_cells = self.road_length * self.lanes
        target_occupied = int(self.density * total_cells)
        occupied = {lane: [False] * self.road_length for lane in range(self.lanes)}

        while sum(sum(row) for row in occupied.values()) < target_occupied:
            lane = random.randint(0, self.lanes - 1)
            pos = random.randint(0, self.road_length - 1)

            r = random.random()
            cumulative = 0
            for (ctype, size, weight) in self.car_types:
                cumulative += weight
                if r <= cumulative:
                    car_type = ctype
                    car_size = size
                    break

            block_free = True
            for i in range(car_size):
                cell = (pos - i) % self.road_length
                if occupied[lane][cell]:
                    block_free = False
                    break

            if not block_free:
                continue

            color = random.choice(self.possible_colors)
            new_car = Car(
                car_id=self.car_counter,
                car_type=car_type,
                size=car_size,
                lane=lane,
                position=pos,
                speed=random.randint(0, self.max_speed),
                color=color
            )
            self.car_counter += 1
            self.cars.append(new_car)

            for i in range(car_size):
                cell = (pos - i) % self.road_length
                occupied[lane][cell] = True

    def build_occupancy_grid(self):
        grid = {lane: [None] * self.road_length for lane in range(self.lanes)}
        for car in self.cars:
            for cell in car.occupied_cells(self.road_length):
                grid[car.lane][cell] = car.id
        return grid

    def get_gap_ahead(self, car, grid):
        gap = 0
        pos = car.position
        for i in range(1, self.road_length):
            cell = (pos + i) % self.road_length
            if grid[car.lane][cell] is None:
                gap += 1
            else:
                if grid[car.lane][cell] == car.id:
                    continue
                else:
                    break
        return gap

    def get_gap_behind(self, lane, pos, grid, ignore_car_id=None):
        gap = 0
        for i in range(1, self.road_length):
            cell = (pos - i) % self.road_length
            occ = grid[lane][cell]
            if occ is None or occ == ignore_car_id:
                gap += 1
            else:
                break
        return gap

    def try_lane_change(self, car, grid):
        if car.speed >= self.max_speed:
            return

        current_gap = self.get_gap_ahead(car, grid)
        candidate_changes = []

        for delta_lane in [-1, 1]:
            new_lane = car.lane + delta_lane
            if new_lane < 0 or new_lane >= self.lanes:
                continue

            block_free = True
            new_cells = [(car.position - i) % self.road_length for i in range(car.size)]
            for cell in new_cells:
                if grid[new_lane][cell] is not None:
                    block_free = False
                    break
            if not block_free:
                continue

            gap_ahead_new = 0
            for i in range(1, self.road_length):
                cell = (car.position + i) % self.road_length
                if grid[new_lane][cell] is None:
                    gap_ahead_new += 1
                else:
                    break

            new_tail = (car.position - car.size + 1) % self.road_length
            gap_behind_new = self.get_gap_behind(new_lane, new_tail, grid)

            if (gap_ahead_new >= current_gap + self.threshold_gap and
                    gap_ahead_new >= self.safe_gap_front and
                    gap_behind_new >= self.safe_gap_rear):
                candidate_changes.append((new_lane, gap_ahead_new))

        if candidate_changes:
            candidate_changes.sort(key=lambda x: x[1], reverse=True)
            new_lane = candidate_changes[0][0]
            car.lane = new_lane

    def update(self):
        grid = self.build_occupancy_grid()

        for car in self.cars:
            self.try_lane_change(car, grid)

        grid = self.build_occupancy_grid()

        for car in self.cars:
            if car.speed < self.max_speed:
                car.speed += 1

            gap = self.get_gap_ahead(car, grid)
            if car.speed > gap:
                car.speed = gap

            if car.speed > 0 and random.random() < self.p_slow:
                car.speed -= 1

        for car in self.cars:
            car.position = (car.position + car.speed) % self.road_length

    def animate(self, interval=200):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_xlim(0, self.road_length)
        ax.set_ylim(-0.5, self.lanes - 0.5)
        ax.set_yticks(range(self.lanes))
        ax.set_yticklabels([f"Lane {i}" for i in range(self.lanes)])
        ax.set_xlabel("Road position (cells)")
        ax.set_title("Motorway Traffic Simulation")

        patches_dict = {}

        for car in self.cars:
            x = car.position - car.size + 1
            rect = patches.Rectangle((x, car.lane - 0.4), car.size, 0.8,
                                     fc=car.color, ec='black', lw=1)
            patches_dict[car.id] = rect
            ax.add_patch(rect)

        def init():
            return list(patches_dict.values())

        def update_frame(frame):
            self.update()
            for car in self.cars:
                x = car.position - car.size + 1
                patches_dict[car.id].set_xy((x, car.lane - 0.4))
                patches_dict[car.id].set_width(car.size)
            ax.set_title(f"Step {frame}")
            return list(patches_dict.values())

        ani = animation.FuncAnimation(fig, update_frame, frames=self.steps,
                                      init_func=init, blit=True, interval=interval,
                                      repeat=False)
        plt.show()

#runs the animation
if __name__ == "__main__":
    sim = TrafficSimulation(
        road_length=200,
        density=0.2,
        max_speed=10,
        p_slow=0.2,
        steps=500
    )
    sim.animate(interval=300)