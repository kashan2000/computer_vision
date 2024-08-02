"""
Module to Calculate All of the Parameters
"""

import measure
from log import logging

MASS = 70

class ParametersCalculator:
    """Class to calculate various parameters based on input data."""

    def __init__(self):
        self.last_measured_data = None  # Initialize cache for last data

    def get_parameters(self, data):
        """Calculate parameters based on input data and frame count."""
        measured_data = {}
        new_frame_count = data['Frame Count']

        if self.last_measured_data is None:
            old_frame_count = 0
        else:
            old_frame_count = self.last_measured_data['Frame Count']
        interval = (new_frame_count - old_frame_count) / 30  # Assuming 30 fps

        try:
            measured_data['look'] = self.calculate_look(data)
            measured_data['tilt'] = self.calculate_tilt(data)
        except Exception as e:
            logging.error("An error occurred during getting head position: %s", str(e))

        try:
            measured_data['Displacements'] = self.calculate_displacement(data)
            measured_data['Velocities'] = self.calculate_velocity(data, interval)
            measured_data['Angular Velocities'] = self.calculate_angular_velocities(data, interval)
        except Exception as e:
            logging.error("An error occurred while getting displacement and velocity: %s", str(e))

        try:
            measured_data['Accelerations'] = self.calculate_accelerations(data, interval)
            measured_data['Angular Accelerations'] = self.calculate_angular_accelerations(data, interval)
            measured_data['Forces'] = self.calculate_forces(data)
        except Exception as e:
            logging.error("An error occurred while calculating acceleration and Forces: %s", str(e))

        data.update(measured_data)
        self.last_measured_data = data
        return data

    def calculate_look(self, data):
        """Calculate the direction of head movement (look) based on keypoint coordinates."""
        required_keys = ('r_shoulder', 'neck', 'nose')

        if all(data.get(key) is not None for key in required_keys):
            look_direction = measure.look_side(data['r_shoulder'], data['neck'], data['nose'])
        else:
            logging.warning("Data for head position calculation not available.")
            look_direction = None
        return look_direction

    def calculate_tilt(self, data):
        """Calculate the tilt of the head based on keypoint coordinates."""
        required_keys = ('r_shoulder', 'l_shoulder', 'neck', 'r_ear', 'l_ear')

        if all(data.get(key) is not None for key in required_keys):
            tilt_angle = measure.head_tilt(data['r_shoulder'], data['l_shoulder'], data['neck'], data['r_ear'], data['l_ear'])
        else:
            logging.warning("Data for head tilt calculation not available.")
            tilt_angle = None
        return tilt_angle

    def calculate_displacement(self, data):
        """Calculate the displacement based on keypoint coordinates."""
        displacements = {}

        if 'Coordinates' in data and 'Coordinates' in self.last_measured_data:  # type: ignore
            for key in ('r_ankle', 'l_ankle', 'r_knee', 'l_knee', 'r_hip', 'l_hip', 'r_shoulder', 'l_shoulder', 'r_elbow', 'l_elbow', 'r_wrist', 'l_wrist', 'ball'):
                position1 = self.last_measured_data['Coordinates'].get(key)  # type: ignore
                position2 = data['Coordinates'].get(key)

                if position1 is not None and position2 is not None:
                    joints_displacement = measure.displacement_joints(position1, position2)
                    ball_displacement = measure.displacement_ball(position1, position2)
                    displacements[key + '_joints_displacement'] = joints_displacement
                    displacements[key + '_ball_displacement'] = ball_displacement
        return displacements

    def calculate_velocity(self, data, interval):
        """Calculate velocities based on input data."""
        velocities = {}

        if 'Displacements' in data and 'Displacements' in self.last_measured_data:  # type: ignore
            for key in data['Displacements']:
                displacement = self.last_measured_data['Displacements'].get(key)  # type: ignore
                if displacement is not None:
                    velocity = displacement / interval
                    velocities[key.replace('displacement', 'velocity')] = velocity
        return velocities

    def calculate_angular_velocities(self, data, interval):
        """Calculate angular velocities based on input data."""
        angular_velocities = {}

        if 'Angles (deg)' in data and 'Angles (deg)' in self.last_measured_data:  # type: ignore
            for key in ('r_elbow_angle', 'l_elbow_angle', 'r_shoulder_angle', 'l_shoulder_angle', 'r_hip_angle', 'l_hip_angle', 'r_knee_angle', 'l_knee_angle'):
                angle1 = self.last_measured_data['Angles (deg)'].get(key)  # type: ignore
                angle2 = data['Angles (deg)'].get(key)
                if angle1 is not None and angle2 is not None:
                    angular_velocity = measure.angular_velocities_angle(angle1, angle2, interval)
                    angular_velocities[key.replace('_angle', '_angular_velocity')] = angular_velocity
        return angular_velocities

    def calculate_accelerations(self, data, interval):
        """Calculate accelerations based on input data."""
        accelerations = {}

        if 'Velocities' in data and 'Velocities' in self.last_measured_data:  # type: ignore
            for key in ('r_elbow_velocity', 'l_elbow_velocity', 'r_shoulder_velocity', 'l_shoulder_velocity', 'r_hip_velocity', 'l_hip_velocity', 'r_knee_velocity', 'l_knee_velocity', 'ball_velocity'):
                velocity_1 = self.last_measured_data['Velocities'].get(key)  # type: ignore
                velocity_2 = data['Velocities'].get(key)
                if velocity_1 is not None and velocity_2 is not None:
                    acceleration = measure.acceleration(velocity_1, velocity_2, interval)
                    accelerations[key.replace('_velocity', '_acceleration')] = acceleration
        return accelerations

    def calculate_angular_accelerations(self, data, interval):
        """Calculate angular accelerations based on input data."""
        angular_accelerations = {}

        if 'Angular Velocities' in data and 'Angular Velocities' in self.last_measured_data:  # type: ignore
            for key in ('r_elbow_angular_velocity', 'l_elbow_angular_velocity', 'r_shoulder_angular_velocity', 'l_shoulder_angular_velocity', 'r_hip_angular_velocity', 'l_hip_angular_velocity', 'r_knee_angular_velocity', 'l_knee_angular_velocity'):
                angular_velocity_1 = self.last_measured_data['Angular Velocities'].get(key)  # type: ignore
                angular_velocity_2 = data['Angular Velocities'].get(key)
                if angular_velocity_1 is not None and angular_velocity_2 is not None:
                    angular_acceleration = measure.angular_accelerations_angular_velocity(angular_velocity_1, angular_velocity_2, interval)
                    angular_accelerations[key.replace('_angular_velocity', '_angular_acceleration')] = angular_acceleration
        return angular_accelerations

    def calculate_forces(self, data):
        """Calculate forces based on angular accelerations."""
        forces = {}

        if 'Angular Accelerations' in data:
            for key in ('r_elbow_angular_acceleration', 'l_elbow_angular_acceleration', 'r_shoulder_angular_acceleration', 'l_shoulder_angular_acceleration', 'r_hip_angular_acceleration', 'l_hip_angular_acceleration', 'r_knee_angular_acceleration', 'l_knee_angular_acceleration'):
                angular_acceleration = data['Angular Accelerations'].get(key)
                if angular_acceleration is not None:
                    force = measure.forces(angular_acceleration, MASS)
                    forces[key.replace('_angular_acceleration', '_angular_force')] = force
        return forces
