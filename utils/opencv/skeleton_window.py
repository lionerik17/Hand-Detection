import cv2
import numpy as np
from .base_window import OpenCVWindow
from utils.hand.hand_model import HAND_CONNECTIONS, HandJoint

class SkeletonWindow(OpenCVWindow):
    """
    Window specialized for displaying a 21-landmark MediaPipe-style skeleton
    reconstructed from FPGA packet data with realistic kinematics.
    """
    def __init__(self, window_name="3D Hand Skeleton", width=600, height=600, mirror=True):
        super().__init__(window_name)
        self.width = width
        self.height = height
        self.scale = 250
        self.mirror = mirror
        
        # Knuckle-facing perspective (degrees)
        self.angle_x = -45
        self.angle_y = 10
        self.angle_z = 0

    def _get_rotation_matrix(self, ax_deg, ay_deg, az_deg):
        ax, ay, az = np.radians([ax_deg, ay_deg, az_deg])
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def rotate_around(self, pt, pivot, axis, angle):
        """Rodrigues rotation of a point around a pivot and axis."""
        v = pt - pivot
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Ensure axis is normalized
        axis = axis / np.linalg.norm(axis)
        v_rot = v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)
        return pivot + v_rot

    def render_from_packet(self, decoded):
        """
        Renders a full 21-landmark skeleton using packet data to transform 
        a neutral 'documentation-style' rest pose with realistic kinematics.
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_grid(canvas)

        if not decoded:
            cv2.putText(canvas, "Awaiting Data...", (self.width//2 - 60, self.height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            return canvas

        # 1. Neutral Rest Pose
        points = np.zeros((21, 3))
        # Wrist
        points[HandJoint.WRIST]      = [0, 0, 0]
        # Thumb
        points[HandJoint.THUMB_CMC]  = [0.15, 0.1, 0.05]
        points[HandJoint.THUMB_MCP]  = [0.35, 0.2, 0.15]
        points[HandJoint.THUMB_IP]   = [0.5, 0.3, 0.25]
        points[HandJoint.THUMB_TIP]  = [0.65, 0.4, 0.35]
        # Index
        points[HandJoint.INDEX_MCP]  = [0.3, 0.6, 0]
        points[HandJoint.INDEX_PIP]  = [0.3, 0.9, 0]
        points[HandJoint.INDEX_DIP]  = [0.3, 1.1, 0]
        points[HandJoint.INDEX_TIP]  = [0.3, 1.25, 0]
        # Middle
        points[HandJoint.MIDDLE_MCP] = [0, 0.7, -0.05]
        points[HandJoint.MIDDLE_PIP] = [0, 1.1, -0.05]
        points[HandJoint.MIDDLE_DIP] = [0, 1.35, -0.05]
        points[HandJoint.MIDDLE_TIP] = [0, 1.5, -0.05]
        # Ring
        points[HandJoint.RING_MCP]   = [-0.3, 0.65, 0]
        points[HandJoint.RING_PIP]   = [-0.3, 1.0, 0]
        points[HandJoint.RING_DIP]   = [-0.3, 1.2, 0]
        points[HandJoint.RING_TIP]   = [-0.3, 1.35, 0]
        # Pinky
        points[HandJoint.PINKY_MCP]  = [-0.6, 0.55, 0.1]
        points[HandJoint.PINKY_PIP]  = [-0.6, 0.8, 0.1]
        points[HandJoint.PINKY_DIP]  = [-0.6, 1.0, 0.1]
        points[HandJoint.PINKY_TIP]  = [-0.6, 1.1, 0.1]

        # 2. Extract Packet Metrics
        # Flexion: 0 straight, high bent. Opposition: dist in mm.
        flexions = [decoded['thumb'], decoded['index'], decoded['middle'], decoded['ring'], decoded['pinky']]
        
        # Normalize Spread (roughly 30-150 mm range to 0-1)
        spread_val = np.clip((decoded['spread'] - 40) / 100.0, 0, 1)
        
        # Normalize Opposition (distance decreases as thumb opposes)
        # Max dist ~120mm, Min dist ~20mm
        opp_factor = np.clip((120 - decoded['opposition']) / 100.0, 0, 1)
        
        wrist_angle = decoded['wrist'] - 90

        # 3. Apply Kinematics
        
        # A. Finger Spread (Rotate chains around the wrist/center)
        for i, mcp_idx in enumerate(HandJoint.get_finger_bases()):
            if mcp_idx == HandJoint.THUMB_CMC: continue   # Thumb has unique opposition logic
            if mcp_idx == HandJoint.MIDDLE_MCP: continue  # Middle is anchor
            
            # Middle is index 2 in [Thumb, Index, Middle, Ring, Pinky]
            # We use (i - 2) to center the spread rotation on Middle
            spread_angle = np.radians((i - 2) * 15 * spread_val)
            for j in range(4): # MCP to TIP
                idx = mcp_idx + j
                points[idx] = self.rotate_around(points[idx], [0, 0, 0], [0, 0, 1], spread_angle)

        # B. Thumb Opposition (Arc across the palm)
        # Rotate thumb chain (2-4) around CMC (1) toward the pinky
        opp_angle = np.radians(opp_factor * 60) # Up to 60 deg across palm
        opp_axis = [0, 1, 0.5] # Diagonal axis for palm-cross
        for i in range(HandJoint.THUMB_MCP, HandJoint.THUMB_TIP + 1):
            points[i] = self.rotate_around(points[i], points[HandJoint.THUMB_CMC], opp_axis, opp_angle)

        # C. Finger Flexion (Curling)
        for i, indices in enumerate(HandJoint.get_finger_chains()):
            f_val = flexions[i]
            f_rad = np.radians(f_val)
            
            # Rotation axis: perpendicular to the finger's plane
            if i == 0:
                # Thumb bend axis is local to its orientation
                thumb_dir = points[HandJoint.THUMB_MCP] - points[HandJoint.THUMB_CMC]
                bend_axis = np.cross(thumb_dir, [0, 0, 1])
            else:
                bend_axis = [1, 0, 0]

            # Rotate chain
            if i == 0:
                # 1. Bend the whole thumb chain around CMC
                points[HandJoint.THUMB_MCP] = self.rotate_around(points[HandJoint.THUMB_MCP],
                                                                 points[HandJoint.THUMB_CMC],
                                                                 bend_axis, f_rad * 0.2)
                points[HandJoint.THUMB_IP]  = self.rotate_around(points[HandJoint.THUMB_IP],
                                                                 points[HandJoint.THUMB_CMC],
                                                                 bend_axis, f_rad * 0.2)
                points[HandJoint.THUMB_TIP] = self.rotate_around(points[HandJoint.THUMB_TIP],
                                                                 points[HandJoint.THUMB_CMC],
                                                                 bend_axis, f_rad * 0.2)
                
                # 2. Bend IP and TIP around the NEW MCP
                points[HandJoint.THUMB_IP]  = self.rotate_around(points[HandJoint.THUMB_IP],
                                                                 points[HandJoint.THUMB_MCP],
                                                                 bend_axis, f_rad * 0.3)
                points[HandJoint.THUMB_TIP] = self.rotate_around(points[HandJoint.THUMB_TIP],
                                                                 points[HandJoint.THUMB_MCP],
                                                                 bend_axis, f_rad * 0.3)
                
                # 3. Bend TIP around the NEW IP
                points[HandJoint.THUMB_TIP] = self.rotate_around(points[HandJoint.THUMB_TIP],
                                                                 points[HandJoint.THUMB_IP],
                                                                 bend_axis, f_rad * 0.5)
            else:
                # Other fingers curl at PIP(6), DIP(7), TIP(8)
                mcp = points[indices[0]]
                points[indices[1]] = self.rotate_around(points[indices[1]], mcp, bend_axis, f_rad * 0.4)
                points[indices[2]] = self.rotate_around(points[indices[2]], mcp, bend_axis, f_rad * 0.7)
                points[indices[3]] = self.rotate_around(points[indices[3]], mcp, bend_axis, f_rad * 1.0)

        # D. Wrist Rotation
        Rw = self._get_rotation_matrix(wrist_angle, 0, 0)
        points = points @ Rw.T

        # 4. View Projection
        R_view = self._get_rotation_matrix(self.angle_x, self.angle_y, self.angle_z)
        rotated = points @ R_view.T
        
        proj = rotated[:, :2] * self.scale
        proj[:, 1] *= -1
        
        if self.mirror:
            proj[:, 0] *= -1
            
        proj += np.array([self.width // 2, self.height // 2 + 100])

        # 5. Rendering
        # Draw Palm Poly
        cv2.fillPoly(canvas, [proj[HandJoint.get_palm_indices()].astype(int)], (40, 40, 40))

        # Z-Sorting
        sorted_connections = sorted(
            HAND_CONNECTIONS,
            key=lambda conn: (rotated[conn[0], 2] + rotated[conn[1], 2]) / 2
        )

        for start_idx, end_idx in sorted_connections:
            pt1 = tuple(proj[start_idx].astype(int))
            pt2 = tuple(proj[end_idx].astype(int))
            avg_z = (rotated[start_idx, 2] + rotated[end_idx, 2]) / 2
            z_factor = np.clip(1.1 + avg_z, 0.6, 1.4)
            color = (int(0 * z_factor), int(255 * z_factor), int(0 * z_factor))
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, (255, 255, 255), -1)

        cv2.putText(canvas, "Hand Skeletonization (Knuckle View)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return canvas

    def _draw_grid(self, canvas):
        grid_color = (25, 25, 25)
        R = self._get_rotation_matrix(self.angle_x, self.angle_y, self.angle_z)
        for i in range(-5, 6):
            p1 = np.array([i * 0.3, 0, -1.5]) @ R.T
            p2 = np.array([i * 0.3, 0, 1.5]) @ R.T
            p3 = np.array([-1.5, 0, i * 0.3]) @ R.T
            p4 = np.array([1.5, 0, i * 0.3]) @ R.T
            for line in [(p1, p2), (p3, p4)]:
                pts = np.array([line[0][:2], line[1][:2]]) * self.scale
                pts[:, 1] *= -1
                if self.mirror: pts[:, 0] *= -1
                pts += np.array([self.width // 2, self.height // 2 + 100])
                cv2.line(canvas, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), grid_color, 1)
