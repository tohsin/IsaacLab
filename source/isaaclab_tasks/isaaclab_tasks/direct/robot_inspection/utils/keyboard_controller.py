import numpy as np
import torch
import weakref
import carb
import omni

class InspectionKeyboardController:
    """A keyboard controller for the robot inspection environment.
    
    Controls:
        W / S: Forward / Backward (Linear Velocity)
        A / D: Left / Right (Angular Velocity)
        Up / Down Arrows: PTZ Tilt Up / Down
        Left / Right Arrows: PTZ Pan Left / Right
    """
    def __init__(self, device="cuda:0", max_vel_speed = 0.5):
        self._device = device
        self.max_vel_speed = max_vel_speed
        
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        
        # note: Use weakref on callbacks to ensure that this object can be deleted
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        
        self._create_key_bindings()
        
        self._pressed_keys = set()
        # Action space: [linear_vel, angular_vel, pan_vel, tilt_vel, zoom]
        self._base_command = np.zeros(5, dtype=np.float32)

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub') and self._keyboard_sub:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def advance(self) -> torch.Tensor:
        """Provides the current action tensor based on keyboard state.
        Shape is (1, 5) to match environment input expectations for 1 environment.
        """
        command = np.zeros(5, dtype=np.float32)
        for key in self._pressed_keys:
            if key in self._INPUT_KEY_MAPPING:
                command += self._INPUT_KEY_MAPPING[key]
        return torch.tensor([command], dtype=torch.float32, device=self._device)

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed_keys.add(event.input.name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(event.input.name)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        # Action mapping: [lin_vel, ang_vel, pan, tilt, zoom]
        self._INPUT_KEY_MAPPING = {
            # Robot Base (Arrow Keys)
            "UP": np.asarray([self.max_vel_speed, 0.0, 0.0, 0.0, 0.0]),
            "DOWN": np.asarray([-self.max_vel_speed, 0.0, 0.0, 0.0, 0.0]),
            "LEFT": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
            "RIGHT": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0]),
            
            # PTZ Camera (A/S/D/X to avoid W entirely)
            "S": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0]),  # Up
            "X": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),   # Down
            "A": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),   # Left
            "D": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0]),  # Right
        }
