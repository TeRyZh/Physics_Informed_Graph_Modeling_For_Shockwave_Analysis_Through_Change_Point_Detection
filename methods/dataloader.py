import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class TrajectoryDataLoader:
    """
    Generic data loader for vehicle trajectory data.
    Can be extended for specific datasets (NGSIM, highD, Zen, etc.)
    """
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None

    def load_data(self) -> pd.DataFrame:
        """Base method for loading data. Should be implemented by child classes."""
        raise NotImplementedError

    def preprocess_data(self) -> pd.DataFrame:
        """Base method for preprocessing. Should be implemented by child classes."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NGSIM Loader (unchanged)
# ---------------------------------------------------------------------------

class NGSIMDataLoader(TrajectoryDataLoader):
    """
    Specific loader for NGSIM trajectory data with enhanced trajectory segmentation.
    """
    def load_data(self, duration: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Load NGSIM format data with optional time window filtering.

        Args:
            duration: Optional tuple of (start_time, end_time) in seconds.
                     If provided, only loads data within this time window.

        Returns:
            DataFrame containing trajectory data
        """
        try:
            column_names = [
                'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
                'Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_Length',
                'v_Width', 'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID',
                'Preceeding', 'Following', 'Space_Hdwy', 'Time_Hdwy'
            ]

            self.raw_data = pd.read_csv(
                self.data_path,
                names=column_names,
                skiprows=1,
                na_values=['nan', 'NaN', 'NA'],
                keep_default_na=True
            )

            self.raw_data = self.raw_data.fillna(0)

            int_columns = ['Vehicle_ID', 'Frame_ID', 'Total_Frames',
                           'Lane_ID', 'Preceeding', 'Following']
            self.raw_data[int_columns] = self.raw_data[int_columns].astype(np.int32)

            # Add time in seconds (NGSIM records at 10 Hz)
            self.raw_data['Time'] = self.raw_data['Frame_ID'] / 10

            # Filter by duration if provided
            if duration:
                start_time, end_time = duration
                self.raw_data = self.raw_data[
                    (self.raw_data['Time'] >= start_time) &
                    (self.raw_data['Time'] <= end_time)
                ]
                print(f"Filtered data for time window: {start_time}s to {end_time}s")

            print(f"Loaded {len(self.raw_data)} records from "
                  f"{len(self.raw_data['Vehicle_ID'].unique())} vehicles.")
            return self.raw_data

        except Exception as e:
            raise Exception(f"Error loading NGSIM data: {str(e)}")

    def preprocess_data(self,
                        min_segment_length: int = 100,
                        duration: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Preprocess NGSIM data with enhanced trajectory segmentation.

        Args:
            min_segment_length: Minimum length for a valid trajectory segment
            duration: Optional tuple of (start_time, end_time) in seconds.

        Returns:
            Preprocessed DataFrame with segmented trajectories
        """
        if self.raw_data is None:
            raise Exception("Data not loaded. Call load_data() first.")

        df = self.raw_data.copy()

        if duration and 'Time' not in df.columns:
            df['Time'] = df['Frame_ID'] / 10
            start_time, end_time = duration
            df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
            print(f"Filtered data for time window: {start_time}s to {end_time}s")
        elif 'Time' not in df.columns:
            df['Time'] = df['Frame_ID'] / 10

        segment_id = 0
        segments = []

        print("Segmenting trajectories...")

        for vehicle_id in df['Vehicle_ID'].unique():
            vehicle_data = df[df['Vehicle_ID'] == vehicle_id].sort_values('Frame_ID')

            if len(vehicle_data) < min_segment_length:
                continue

            lane_changes = vehicle_data['Lane_ID'].diff().fillna(0) != 0
            change_points = vehicle_data.index[lane_changes].tolist()
            segment_points = ([vehicle_data.index[0]] +
                              change_points +
                              [vehicle_data.index[-1]])

            for i in range(len(segment_points) - 1):
                start_idx = segment_points[i]
                end_idx = segment_points[i + 1]
                segment = vehicle_data.loc[start_idx:end_idx].copy()

                if len(segment) >= min_segment_length:
                    segment['Distance'] = segment['Local_Y']
                    segment['Segment_ID'] = segment_id
                    segment['Original_Vehicle_ID'] = vehicle_id
                    segments.append(segment)
                    segment_id += 1

        if not segments:
            raise Exception("No valid trajectory segments found after processing.")

        self.processed_data = pd.concat(segments).sort_values('Time')

        time_range = (f" (Time window: {duration[0]}s to {duration[1]}s)"
                      if duration else "")
        print(f"\nProcessing summary{time_range}:")
        print(f"Original vehicles: {len(df['Vehicle_ID'].unique())}")
        print(f"Generated segments: {segment_id}")
        print(f"Average segment length: "
              f"{len(self.processed_data) / segment_id:.1f} frames")

        return self.processed_data

    def get_trajectories(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract all individual trajectory segments."""
        if self.processed_data is None:
            raise Exception("Data not processed. Call preprocess_data() first.")

        trajectories = {}
        for seg_id in self.processed_data['Segment_ID'].unique():
            seg = self.processed_data[self.processed_data['Segment_ID'] == seg_id]
            trajectories[seg_id] = {
                'time':                seg['Time'].values,
                'distance':            seg['Distance'].values,
                'velocity':            seg['v_Vel'].values,
                'acceleration':        seg['v_Acc'].values,
                'lane_id':             seg['Lane_ID'].iloc[0],
                'original_vehicle_id': seg['Original_Vehicle_ID'].iloc[0]
            }
        return trajectories

    def get_lane_trajectories(self, lane_id: int) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract trajectory segments for a specific lane."""
        if self.processed_data is None:
            raise Exception("Data not processed. Call preprocess_data() first.")

        lane_data = self.processed_data[self.processed_data['Lane_ID'] == lane_id]

        if len(lane_data) == 0:
            print(f"No trajectories found for lane {lane_id}")
            return {}

        trajectories = {}
        for seg_id in lane_data['Segment_ID'].unique():
            seg = self.processed_data[self.processed_data['Segment_ID'] == seg_id]
            trajectories[seg_id] = {
                'time':                seg['Time'].values,
                'distance':            seg['Distance'].values,
                'velocity':            seg['v_Vel'].values,
                'acceleration':        seg['v_Acc'].values,
                'original_vehicle_id': seg['Original_Vehicle_ID'].iloc[0],
                'lane_id':             lane_id
            }

        print(f"Found {len(trajectories)} trajectories for lane {lane_id}")
        return trajectories


# ---------------------------------------------------------------------------
# Zen Dataset Loader
# ---------------------------------------------------------------------------

class ZenDataLoader(TrajectoryDataLoader):
    """
    Loader for the Zen (Japanese expressway) trajectory dataset.

    CSV format (no header, 10 columns):
        vehicle_id   : int   – Vehicle ID
        datetime     : int   – HHMMSSFFF  (FFF increments by 100 per 0.1 s → 10 Hz)
        vehicle_type : int   – 1: normal vehicle, 2: large vehicle (bus/truck)
        velocity     : float – km/h
        traffic_lane : int   – 1: driving, 2: passing, 3: entrance
        longitude    : float – WGS84 degrees
        latitude     : float – WGS84 degrees
        kilopost     : float – metres from expressway start (used as Distance)
        vehicle_length: float – metres
        detected_flag: int   – 1: detected, 0: interpolated

    Key differences from NGSIM:
    - Velocity in km/h  → converted to m/s internally
    - Time encoded as HHMMSSFFF; parsed to seconds-of-day
    - Distance is kilopost (m along the road), not a lateral position
    - No explicit acceleration column; computed from velocity differences
    - Lane changes detected the same way as NGSIM (Lane_ID diff)
    """

    # Zen column names after loading
    _COLUMNS = [
        'Vehicle_ID', 'datetime_raw', 'vehicle_type', 'velocity_kmh',
        'Lane_ID', 'longitude', 'latitude', 'kilopost',
        'vehicle_length', 'detected_flag'
    ]

    @staticmethod
    def _parse_datetime_to_seconds(dt_series: pd.Series) -> pd.Series:
        """
        Convert integer HHMMSSFFF → float seconds-of-day.
        FFF step of 100 corresponds to 0.1 s (10 Hz recording).
        """
        s = dt_series.astype(str).str.zfill(9)
        hh  = s.str[:2].astype(float)
        mm  = s.str[2:4].astype(float)
        ss  = s.str[4:6].astype(float)
        fff = s.str[6:].astype(float)   # 000–900, step 100
        return hh * 3600.0 + mm * 60.0 + ss + fff / 1000.0

    def load_data(self,
                  duration: Optional[Tuple[float, float]] = None,
                  exclude_large_vehicles: bool = False) -> pd.DataFrame:
        """
        Load Zen dataset CSV.

        Args:
            duration: Optional (start_sec, end_sec) window relative to the
                      earliest timestamp in the file (i.e. elapsed seconds).
                      Pass None to load the full file.
            exclude_large_vehicles: If True, drop vehicle_type == 2 records.

        Returns:
            DataFrame with a 'Time' column (elapsed seconds from file start),
            'v_Vel' (m/s), 'v_Acc' (m/s²), and 'Distance' (kilopost, metres).
        """
        try:
            self.raw_data = pd.read_csv(
                self.data_path,
                header=None,
                names=self._COLUMNS,
                na_values=['nan', 'NaN', 'NA'],
                keep_default_na=True
            ).fillna(0)

            # Parse absolute time (seconds of day)
            self.raw_data['time_abs'] = self._parse_datetime_to_seconds(
                self.raw_data['datetime_raw']
            )

            # Elapsed time from file start
            t0 = self.raw_data['time_abs'].min()
            self.raw_data['Time'] = self.raw_data['time_abs'] - t0

            # Convert velocity km/h → m/s  (matches NGSIM's ft/s unit convention
            # only in spirit; keep m/s since Zen uses metric distances)
            self.raw_data['v_Vel'] = self.raw_data['velocity_kmh'] / 3.6

            # Acceleration (m/s²) computed per-vehicle via forward difference
            self.raw_data = self.raw_data.sort_values(['Vehicle_ID', 'Time'])
            dt = 0.1  # 10 Hz → Δt = 0.1 s
            self.raw_data['v_Acc'] = (
                self.raw_data.groupby('Vehicle_ID')['v_Vel']
                .diff()
                .fillna(0) / dt
            )

            # Distance = kilopost (metres)
            self.raw_data['Distance'] = self.raw_data['kilopost']

            if exclude_large_vehicles:
                before = len(self.raw_data['Vehicle_ID'].unique())
                self.raw_data = self.raw_data[self.raw_data['vehicle_type'] != 2]
                after = len(self.raw_data['Vehicle_ID'].unique())
                print(f"Excluded large vehicles: {before - after} vehicles removed.")

            # Duration filter (elapsed seconds)
            if duration:
                start_t, end_t = duration
                self.raw_data = self.raw_data[
                    (self.raw_data['Time'] >= start_t) &
                    (self.raw_data['Time'] <= end_t)
                ]
                print(f"Filtered data for time window: {start_t}s to {end_t}s")

            n_veh = self.raw_data['Vehicle_ID'].nunique()
            print(f"Loaded {len(self.raw_data)} records from {n_veh} vehicles.")
            return self.raw_data

        except Exception as e:
            raise Exception(f"Error loading Zen data: {str(e)}")

    def preprocess_data(self,
                        min_segment_length: int = 50,
                        duration: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Segment Zen trajectories by lane changes, mirroring NGSIM logic.

        Args:
            min_segment_length: Minimum frames for a valid segment.
                                 Default is 50 (5 s at 10 Hz) – lower than NGSIM
                                 because the Zen dataset covers shorter windows.
            duration: Optional elapsed-time filter applied before segmentation.

        Returns:
            Preprocessed DataFrame with 'Segment_ID' and 'Original_Vehicle_ID'.
        """
        if self.raw_data is None:
            raise Exception("Data not loaded. Call load_data() first.")

        df = self.raw_data.copy()

        if duration:
            start_t, end_t = duration
            df = df[(df['Time'] >= start_t) & (df['Time'] <= end_t)]
            print(f"Filtered data for time window: {start_t}s to {end_t}s")

        segment_id = 0
        segments = []

        print("Segmenting Zen trajectories...")

        for vehicle_id in df['Vehicle_ID'].unique():
            vdata = df[df['Vehicle_ID'] == vehicle_id].sort_values('Time')

            if len(vdata) < min_segment_length:
                continue

            lane_changes = vdata['Lane_ID'].diff().fillna(0) != 0
            change_points = vdata.index[lane_changes].tolist()
            seg_points = ([vdata.index[0]] +
                          change_points +
                          [vdata.index[-1]])

            for i in range(len(seg_points) - 1):
                seg = vdata.loc[seg_points[i]:seg_points[i + 1]].copy()
                if len(seg) >= min_segment_length:
                    seg['Segment_ID'] = segment_id
                    seg['Original_Vehicle_ID'] = vehicle_id
                    segments.append(seg)
                    segment_id += 1

        if not segments:
            raise Exception("No valid Zen trajectory segments found.")

        self.processed_data = pd.concat(segments).sort_values('Time')

        print(f"\nZen processing summary:")
        print(f"Original vehicles: {df['Vehicle_ID'].nunique()}")
        print(f"Generated segments: {segment_id}")
        if segment_id > 0:
            print(f"Average segment length: "
                  f"{len(self.processed_data) / segment_id:.1f} frames")

        return self.processed_data

    def get_trajectories(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract all individual Zen trajectory segments."""
        if self.processed_data is None:
            raise Exception("Data not processed. Call preprocess_data() first.")

        trajectories = {}
        for seg_id in self.processed_data['Segment_ID'].unique():
            seg = self.processed_data[self.processed_data['Segment_ID'] == seg_id]
            trajectories[int(seg_id)] = {
                'time':                seg['Time'].values,
                'distance':            seg['Distance'].values,       # kilopost (m)
                'velocity':            seg['v_Vel'].values,           # m/s
                'acceleration':        seg['v_Acc'].values,           # m/s²
                'lane_id':             int(seg['Lane_ID'].iloc[0]),
                'original_vehicle_id': int(seg['Original_Vehicle_ID'].iloc[0])
            }
        return trajectories

    def get_lane_trajectories(self, lane_id: int) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract Zen trajectory segments for a specific lane."""
        if self.processed_data is None:
            raise Exception("Data not processed. Call preprocess_data() first.")

        lane_data = self.processed_data[self.processed_data['Lane_ID'] == lane_id]
        if len(lane_data) == 0:
            print(f"No Zen trajectories found for lane {lane_id}")
            return {}

        trajectories = {}
        for seg_id in lane_data['Segment_ID'].unique():
            seg = self.processed_data[self.processed_data['Segment_ID'] == seg_id]
            trajectories[int(seg_id)] = {
                'time':                seg['Time'].values,
                'distance':            seg['Distance'].values,
                'velocity':            seg['v_Vel'].values,
                'acceleration':        seg['v_Acc'].values,
                'original_vehicle_id': int(seg['Original_Vehicle_ID'].iloc[0]),
                'lane_id':             lane_id
            }

        print(f"Found {len(trajectories)} Zen trajectories for lane {lane_id}")
        return trajectories


# ---------------------------------------------------------------------------
# highD Loader
# ---------------------------------------------------------------------------

class HighDDataLoader(TrajectoryDataLoader):
    """
    Loader for the highD dataset (German highway, aerial drone video).

    Expected CSV structure (standard highD track CSV, one row per frame):
        id, frame, totalFrames, width, height,
        xVelocity, yVelocity, xAcceleration, yAcceleration,
        frontSightDistance, backSightDistance,
        dhw, thw, ttc, precedingXVelocity,
        precedingId, followingId,
        leftPrecedingId, leftAlongsideId, leftFollowingId,
        rightPrecedingId, rightAlongsideId, rightFollowingId,
        laneId, class, x, y

    Key properties:
    - Frame rate: 25 Hz  → Time = frame / 25
    - Velocity: m/s (xVelocity along road; yVelocity lateral)
    - Position: x (along road, metres), y (lateral, metres)
    - class: 'Car' or 'Truck'
    - Lane IDs are track-file specific; upper/lower road sections have
      different lane numbering (see highD recording metadata CSV).
    """

    _FRAME_RATE = 25.0  # Hz

    def load_data(self,
                  duration: Optional[Tuple[float, float]] = None,
                  exclude_trucks: bool = False) -> pd.DataFrame:
        """
        Load a highD track CSV file.

        Args:
            duration: Optional (start_sec, end_sec) elapsed-time filter.
            exclude_trucks: If True, drop records where class == 'Truck'.

        Returns:
            DataFrame with standardised columns including 'Time', 'v_Vel',
            'v_Acc', 'Distance', 'Lane_ID', 'Vehicle_ID'.
        """
        try:
            self.raw_data = pd.read_csv(
                self.data_path,
                na_values=['nan', 'NaN', 'NA'],
                keep_default_na=True
            ).fillna(0)

            # Normalise column names (highD uses camelCase)
            self.raw_data.rename(columns={
                'id':              'Vehicle_ID',
                'frame':           'Frame_ID',
                'totalFrames':     'Total_Frames',
                'laneId':          'Lane_ID',
                'class':           'v_Class',
                'x':               'Local_X',
                'y':               'Local_Y',
                'xVelocity':       'v_Vel_raw',
                'yVelocity':       'v_Vel_lat',
                'xAcceleration':   'v_Acc_raw',
                'yAcceleration':   'v_Acc_lat',
            }, inplace=True)

            # Elapsed time (seconds) from first frame
            self.raw_data['Time'] = (
                (self.raw_data['Frame_ID'] - self.raw_data['Frame_ID'].min())
                / self._FRAME_RATE
            )

            # Longitudinal speed (m/s); take absolute value so direction is
            # captured by sign convention in the graph clusterer if needed.
            # highD records negative xVelocity for the upper road direction.
            self.raw_data['v_Vel'] = self.raw_data['v_Vel_raw']   # keep sign
            self.raw_data['v_Acc'] = self.raw_data['v_Acc_raw']

            # Distance = longitudinal position x (metres)
            self.raw_data['Distance'] = self.raw_data['Local_X']

            if exclude_trucks:
                before = self.raw_data['Vehicle_ID'].nunique()
                self.raw_data = self.raw_data[
                    self.raw_data['v_Class'].str.strip().str.lower() != 'truck'
                ]
                after = self.raw_data['Vehicle_ID'].nunique()
                print(f"Excluded trucks: {before - after} vehicles removed.")

            if duration:
                start_t, end_t = duration
                self.raw_data = self.raw_data[
                    (self.raw_data['Time'] >= start_t) &
                    (self.raw_data['Time'] <= end_t)
                ]
                print(f"Filtered data for time window: {start_t}s to {end_t}s")

            n_veh = self.raw_data['Vehicle_ID'].nunique()
            print(f"Loaded {len(self.raw_data)} records from {n_veh} vehicles.")
            return self.raw_data

        except Exception as e:
            raise Exception(f"Error loading highD data: {str(e)}")

    def preprocess_data(self,
                        min_segment_length: int = 100,
                        duration: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Segment highD trajectories by lane changes.

        Args:
            min_segment_length: Minimum frames per valid segment
                                 (100 frames = 4 s at 25 Hz).
            duration: Optional elapsed-time filter.

        Returns:
            Preprocessed DataFrame with 'Segment_ID' and 'Original_Vehicle_ID'.
        """
        if self.raw_data is None:
            raise Exception("Data not loaded. Call load_data() first.")

        df = self.raw_data.copy()

        if duration:
            start_t, end_t = duration
            df = df[(df['Time'] >= start_t) & (df['Time'] <= end_t)]
            print(f"Filtered data for time window: {start_t}s to {end_t}s")

        segment_id = 0
        segments = []

        print("Segmenting highD trajectories...")

        for vehicle_id in df['Vehicle_ID'].unique():
            vdata = df[df['Vehicle_ID'] == vehicle_id].sort_values('Frame_ID')

            if len(vdata) < min_segment_length:
                continue

            lane_changes = vdata['Lane_ID'].diff().fillna(0) != 0
            change_points = vdata.index[lane_changes].tolist()
            seg_points = ([vdata.index[0]] +
                          change_points +
                          [vdata.index[-1]])

            for i in range(len(seg_points) - 1):
                seg = vdata.loc[seg_points[i]:seg_points[i + 1]].copy()
                if len(seg) >= min_segment_length:
                    seg['Segment_ID'] = segment_id
                    seg['Original_Vehicle_ID'] = vehicle_id
                    segments.append(seg)
                    segment_id += 1

        if not segments:
            raise Exception("No valid highD trajectory segments found.")

        self.processed_data = pd.concat(segments).sort_values('Time')

        print(f"\nhighD processing summary:")
        print(f"Original vehicles: {df['Vehicle_ID'].nunique()}")
        print(f"Generated segments: {segment_id}")
        if segment_id > 0:
            print(f"Average segment length: "
                  f"{len(self.processed_data) / segment_id:.1f} frames")

        return self.processed_data

    def get_trajectories(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract all individual highD trajectory segments."""
        if self.processed_data is None:
            raise Exception("Data not processed. Call preprocess_data() first.")

        trajectories = {}
        for seg_id in self.processed_data['Segment_ID'].unique():
            seg = self.processed_data[self.processed_data['Segment_ID'] == seg_id]
            trajectories[int(seg_id)] = {
                'time':                seg['Time'].values,
                'distance':            seg['Distance'].values,   # x, metres
                'velocity':            seg['v_Vel'].values,       # m/s (signed)
                'acceleration':        seg['v_Acc'].values,       # m/s²
                'lane_id':             int(seg['Lane_ID'].iloc[0]),
                'original_vehicle_id': int(seg['Original_Vehicle_ID'].iloc[0])
            }
        return trajectories

    def get_lane_trajectories(self, lane_id: int) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract highD trajectory segments for a specific lane."""
        if self.processed_data is None:
            raise Exception("Data not processed. Call preprocess_data() first.")

        lane_data = self.processed_data[self.processed_data['Lane_ID'] == lane_id]
        if len(lane_data) == 0:
            print(f"No highD trajectories found for lane {lane_id}")
            return {}

        trajectories = {}
        for seg_id in lane_data['Segment_ID'].unique():
            seg = self.processed_data[self.processed_data['Segment_ID'] == seg_id]
            trajectories[int(seg_id)] = {
                'time':                seg['Time'].values,
                'distance':            seg['Distance'].values,
                'velocity':            seg['v_Vel'].values,
                'acceleration':        seg['v_Acc'].values,
                'original_vehicle_id': int(seg['Original_Vehicle_ID'].iloc[0]),
                'lane_id':             lane_id
            }

        print(f"Found {len(trajectories)} highD trajectories for lane {lane_id}")
        return trajectories


# ---------------------------------------------------------------------------
# Custom / placeholder loader
# ---------------------------------------------------------------------------

class CustomDataLoader(TrajectoryDataLoader):
    """Template for creating loaders for other trajectory datasets."""

    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def preprocess_data(self) -> pd.DataFrame:
        raise NotImplementedError
