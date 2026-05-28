import cv2
import numpy as np
from .base_window import OpenCVWindow
from utils.hand.hand_model import HAND_CONNECTIONS, HandJoint
from utils.pose.pose_model import PoseJoint

class SkeletonWindow(OpenCVWindow):
    """
    Window specialized for displaying a 21-landmark MediaPipe-style skeleton
    plus arm (elbow and shoulder) reconstructed from FPGA packet data.
    """
    def __init__(self, window_name="3D Hand Skeleton", width=800, height=800, mirror=True):
        super().__init__(window_name)
        self.width = width
        self.height = height
        self.scale = 180
        self.mirror = mirror
        self.offset_y = 150
        self.offset_x = 150

        # Perspective angles (degrees)
        self.angle_x = -15
        self.angle_y = 30
        self.angle_z = 90

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
        axis = axis / np.linalg.norm(axis)
        v_rot = v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)
        return pivot + v_rot

    def render_from_packet(self, decoded):
        """
        Renders a full 21-landmark skeleton plus arm using packet data.
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_grid(canvas)

        if not decoded:
            cv2.putText(canvas, "Awaiting Data...", (self.width//2 - 60, self.height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            return canvas

        # 1. Neutral Rest Pose (Wrist rooted at [0,0,0], Fingers UP)
        points = np.zeros((23, 3))
        
        # Hand joints (same as original, relative to Wrist)
        points[HandJoint.WRIST]      = [0, 0, 0]
        points[HandJoint.THUMB_CMC]  = [0.15, 0.1, 0.05]
        points[HandJoint.THUMB_MCP]  = [0.35, 0.2, 0.15]
        points[HandJoint.THUMB_IP]   = [0.5, 0.3, 0.25]
        points[HandJoint.THUMB_TIP]  = [0.65, 0.4, 0.35]
        points[HandJoint.INDEX_MCP]  = [0.3, 0.6, 0]
        points[HandJoint.INDEX_PIP]  = [0.3, 0.9, 0]
        points[HandJoint.INDEX_DIP]  = [0.3, 1.1, 0]
        points[HandJoint.INDEX_TIP]  = [0.3, 1.25, 0]
        points[HandJoint.MIDDLE_MCP] = [0, 0.7, -0.05]
        points[HandJoint.MIDDLE_PIP] = [0, 1.1, -0.05]
        points[HandJoint.MIDDLE_DIP] = [0, 1.35, -0.05]
        points[HandJoint.MIDDLE_TIP] = [0, 1.5, -0.05]
        points[HandJoint.RING_MCP]   = [-0.3, 0.65, 0]
        points[HandJoint.RING_PIP]   = [-0.3, 1.0, 0]
        points[HandJoint.RING_DIP]   = [-0.3, 1.2, 0]
        points[HandJoint.RING_TIP]   = [-0.3, 1.35, 0]
        points[HandJoint.PINKY_MCP]  = [-0.6, 0.55, 0.1]
        points[HandJoint.PINKY_PIP]  = [-0.6, 0.8, 0.1]
        points[HandJoint.PINKY_DIP]  = [-0.6, 1.0, 0.1]
        points[HandJoint.PINKY_TIP]  = [-0.6, 1.1, 0.1]

        # Arm joints (extending DOWN from wrist)
        points[PoseJoint.ELBOW]       = [0, -1.0, 0]
        points[PoseJoint.SHOULDER]    = [0, -2.0, 0]

        # 2. Extract Packet Metrics
        # Remap for simulation/real data
        flexions = [decoded.get('thumb', 0), decoded.get('index', 0), decoded.get('middle', 0), 
                    decoded.get('ring', 0), decoded.get('pinky', 0)]
        spread_val = np.clip((abs(flexions[1] - flexions[4])) / 90.0, 0, 1)
        opp_factor = np.clip(decoded.get('thumb_palm', 0) / 180.0, 0, 1)
        wrist_angle = decoded.get('wrist', 90) - 90
        elbow_angle_deg = decoded.get('elbow', 180)

        # 3. Apply Kinematics

        # A. Wrist Rotation (Rotate Hand joints)
        Rw = self._get_rotation_matrix(wrist_angle, 0, 0)
        points[:21] = points[:21] @ Rw.T

        # B. Finger Spread
        for i, mcp_idx in enumerate(HandJoint.get_finger_bases()):
            if mcp_idx == HandJoint.THUMB_CMC: continue
            if mcp_idx == HandJoint.MIDDLE_MCP: continue
            spread_angle = np.radians((i - 2) * 15 * spread_val)
            for j in range(4):
                idx = mcp_idx + j
                points[idx] = self.rotate_around(points[idx], [0, 0, 0], [0, 0, 1], spread_angle)

        # C. Thumb Opposition
        opp_angle = np.radians(opp_factor * 60)
        opp_axis = [0, 1, 0.5]
        for i in range(HandJoint.THUMB_MCP, HandJoint.THUMB_TIP + 1):
            points[i] = self.rotate_around(points[i], points[HandJoint.THUMB_CMC], opp_axis, opp_angle)

        # D. Finger Flexion
        for i, indices in enumerate(HandJoint.get_finger_chains()):
            f_val = flexions[i]
            f_rad = np.radians(f_val)
            if i == 0:
                thumb_dir = points[HandJoint.THUMB_MCP] - points[HandJoint.THUMB_CMC]
                bend_axis = np.cross(thumb_dir, [0, 0, 1])
                for idx in [HandJoint.THUMB_MCP, HandJoint.THUMB_IP, HandJoint.THUMB_TIP]:
                    points[idx] = self.rotate_around(points[idx], points[HandJoint.THUMB_CMC], bend_axis, f_rad * 0.2)
                for idx in [HandJoint.THUMB_IP, HandJoint.THUMB_TIP]:
                    points[idx] = self.rotate_around(points[idx], points[HandJoint.THUMB_MCP], bend_axis, f_rad * 0.3)
                points[HandJoint.THUMB_TIP] = self.rotate_around(points[HandJoint.THUMB_TIP], points[HandJoint.THUMB_IP], bend_axis, f_rad * 0.5)
            else:
                mcp = points[indices[0]]
                bend_axis = [1, 0, 0]
                points[indices[1]] = self.rotate_around(points[indices[1]], mcp, bend_axis, f_rad * 0.4)
                points[indices[2]] = self.rotate_around(points[indices[2]], mcp, bend_axis, f_rad * 0.7)
                points[indices[3]] = self.rotate_around(points[indices[3]], mcp, bend_axis, f_rad * 1.0)

        # E. Arm Kinematics (Pivot Hand around Elbow)
        # The upper arm (Shoulder -> Elbow) stays fixed. 
        # We rotate all 21 hand points around the elbow.
        flexion_rad = np.radians(180 - elbow_angle_deg)
        for i in range(21):
            # We use -flexion_rad to ensure the hand bends toward the shoulder 
            # in the correct visual direction.
            points[i] = self.rotate_around(points[i], points[PoseJoint.ELBOW], [0, 0, 1], -flexion_rad)
        
        # 4. View Projection
        R_view = self._get_rotation_matrix(self.angle_x, self.angle_y, self.angle_z)
        rotated = points @ R_view.T
        proj = rotated[:, :2] * self.scale
        proj[:, 1] *= -1
        if self.mirror: proj[:, 0] *= -1
        
        # Center: Fixed-extent centering to prevent shaking
        # We assume a fixed reach to keep the view stable
        # Wrist is at [0,0,0], Hand extends ~1.5 units, Arm extends ~2.0 units
        # We offset the whole assembly so it's balanced around the window center
        offset = np.array([self.width // 2 + self.offset_x, self.height // 2 + self.offset_y])
        proj += offset

        # 5. Rendering
        cv2.fillPoly(canvas, [proj[HandJoint.get_palm_indices()].astype(int)], (40, 40, 40))

        # Draw Arm (Shoulder -> Elbow -> Wrist)
        cv2.line(canvas, tuple(proj[PoseJoint.SHOULDER].astype(int)), tuple(proj[PoseJoint.ELBOW].astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, tuple(proj[PoseJoint.ELBOW].astype(int)), tuple(proj[HandJoint.WRIST].astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(proj[PoseJoint.SHOULDER].astype(int)), 3, (0, 200, 255), -1)
        cv2.circle(canvas, tuple(proj[PoseJoint.ELBOW].astype(int)), 3, (0, 200, 255), -1)

        # Draw Hand
        sorted_connections = sorted(HAND_CONNECTIONS, key=lambda conn: (rotated[conn[0], 2] + rotated[conn[1], 2]) / 2)
        for start_idx, end_idx in sorted_connections:
            pt1 = tuple(proj[start_idx].astype(int))
            pt2 = tuple(proj[end_idx].astype(int))
            avg_z = (rotated[start_idx, 2] + rotated[end_idx, 2]) / 2
            z_factor = np.clip(1.1 + avg_z, 0.6, 1.4)
            color = (int(0 * z_factor), int(255 * z_factor), int(0 * z_factor))
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, (255, 255, 255), -1)

        cv2.putText(canvas, "3D Skeleton with Arm", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
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
                pts += np.array([self.width // 2 + self.offset_x, self.height // 2 + self.offset_y])
                cv2.line(canvas, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), grid_color, 1)
