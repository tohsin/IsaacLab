import torch
import numpy as np
import matplotlib.pyplot as plt

class NormalizeReward:
    def __init__(self, gamma=0.99, epsilon=1e-8, m_value = 1, device=None):
        self.return_rms = RunningMeanStd(shape=(), device=device)
        self.count = 0
        self.epsilon = epsilon
        self.m_value = m_value
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def __call__(self, reward):
        # Update running mean and variance using Welford's algorithm
        if self.update_running_mean:
            self.return_rms.update(reward)
        if torch.isnan(self.return_rms.var).any():
            # print("NaN detected in reward variance!")
            # Handle NaN variance, e.g., by resetting it
            self.return_rms.var = torch.ones_like(self.return_rms.var)
        normalized_reward = reward/ (torch.sqrt(self.return_rms.var) + self.epsilon)
        if torch.isnan(normalized_reward).any():
            print("NaN detected in normalized_reward!")
            print(f"Original reward: {reward}")
            print(f"Reward variance: {self.return_rms.var}")
        normalized_reward = torch.clamp(normalized_reward, -self.m_value, self.m_value)
        return normalized_reward
    

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon
        self.device = device

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.update_mean_var_count_from_moments(
            batch_mean, batch_var, batch_count
        )

    def update_mean_var_count_from_moments(
        self, batch_mean, batch_var, batch_count
    ):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean, self.var, self.count = new_mean, new_var, new_count
        
class RewardScaler:
    def __init__(self, epsilon=1e-8, is_neg_pos1=True):
        """
        Args:
            is_neg_pos1 (bool): If True, scale to [-1, 1]. Otherwise, scale to [0, 1].
            epsilon (float): A small value to prevent division by zero.
        """
        self.running_reward_min = float('inf')
        self.running_reward_max = float('-inf')
        self.epsilon = epsilon
        self.is_neg_pos1 = is_neg_pos1

    def update(self, reward):
        # Update running min and max rewards
        current_batch_min = torch.min(reward).item()
        current_batch_max = torch.max(reward).item()

        if current_batch_min < self.running_reward_min:
            self.running_reward_min = current_batch_min
        if current_batch_max > self.running_reward_max:
            self.running_reward_max = current_batch_max


    def __call__(self, reward : torch.Tensor)-> torch.Tensor:
        self.update(reward)
        # Normalize reward to be within [min_reward, max_reward]
        reward_range = self.running_reward_max - self.running_reward_min

        # 3. Normalize the reward. Uncomment the one you want to use.

        if self.is_neg_pos1:
            normalized_reward = 2 * (reward - self.running_reward_min) / (reward_range + self.epsilon) - 1

        else:
            normalized_reward = (reward - self.running_reward_min) / (reward_range + self.epsilon)
        # normalized_reward = (total_reward - self.running_reward_min) / (reward_range + epsilon)

        # If reward_range is zero, all rewards have been identical. Set normalized reward to 0.
        if reward_range < self.epsilon:
            normalized_reward.fill_(0.0)

        return normalized_reward.clone().unsqueeze(1)
    

def visualise_faces(self, face_ids_to_show):
        """Visualize the discovered faces in the scene using Matplotlib."""
        if face_ids_to_show is None:
            return

        # --- Matplotlib window setup (only runs once) ---
        # If the figure does not exist, create it.
        if not hasattr(self, 'fig_face'):
            plt.ion()  # Turn on interactive mode
            self.fig_face, self.ax_face = plt.subplots()
            self.fig_face.canvas.manager.set_window_title("Face ID Detection")


        # --- Image and data processing (same as before) ---
        face_ids = face_ids_to_show.cpu().numpy().squeeze()
        valid_mask = face_ids != -1

        # Create the visualization image (black with green highlights for faces)
        face_vis = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
        face_vis[valid_mask] = [0, 255, 0]  # Green for detected faces

        # Count valid detections
        valid_count = np.sum(valid_mask)

        # --- Display with Matplotlib ---
        self.ax_face.clear()  # Clear the previous frame
        self.ax_face.imshow(face_vis)  # Display the new image

        # Add text to the image
        self.ax_face.text(5, 15, "Face IDs (Green=Hit)", color='white', fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.5))
        self.ax_face.text(5, 30, f"Valid pixels: {valid_count}", color='white', fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.5))
        
        # We don't want axis ticks for an image display
        self.ax_face.set_xticks([])
        self.ax_face.set_yticks([])

        # Redraw the canvas to show the updates
        self.fig_face.canvas.draw()
        self.fig_face.canvas.flush_events()


def _show_face_ids_(
            self,
            face_ids: torch.Tensor,
            target_mask: torch.Tensor,
            env_id: int = 0,
            win: str = "face_ids_debug",
            scale: int = 6,
            max_ids_in_text: int = 12,):
        import cv2
        x = face_ids[env_id]
        if x.ndim == 3:
            x = x[..., 0]
        x = x.to(torch.int64)

        valid = x >= 0

        if target_mask is not None:
            tm = target_mask[env_id]
            if tm.ndim == 3:
                tm = tm[..., 0]
            tm = tm.bool()
        else:
            tm = None

        # Build RGB image (H,W,3) in torch
        H, W = x.shape
        img = torch.zeros((H, W, 3), device=x.device, dtype=torch.uint8)

        if tm is None:
            img[valid] = torch.tensor([0, 255, 0], device=x.device, dtype=torch.uint8)
            ids = x[valid]
        else:
            img[tm & valid] = torch.tensor([0, 255, 0], device=x.device, dtype=torch.uint8)
            ids = x[tm & valid]

        unique_ids = torch.unique(ids) if ids.numel() > 0 else torch.empty((0,), device=x.device, dtype=torch.int64)
        n_unique = int(unique_ids.numel())
        shown = unique_ids[:max_ids_in_text].tolist()

        # To numpy for OpenCV
        img_np = img.detach().cpu().numpy()  # RGB uint8

        # Overlay text (OpenCV uses BGR color tuples but we're drawing on RGB array; color choice still fine visually)
        cv2.putText(img_np, f"env={env_id}  unique={n_unique}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_np, f"ids[:{max_ids_in_text}]={shown}", (5, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Scale up so it’s visible
        if scale != 1:
            img_np = cv2.resize(img_np, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(win, img_np)
        key = cv2.waitKey(1)  # required for live refresh [web:31]
        return key

