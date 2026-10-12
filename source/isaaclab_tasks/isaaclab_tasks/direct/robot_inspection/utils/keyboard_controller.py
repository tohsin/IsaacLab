import numpy as np
import torch
import weakref
import carb
import omni

class InspectionKeyboardController:
    """A keyboard controller for the robot inspection environment.
    
    Controls:
        Up / Down Arrows: Forward / Backward
        Left / Right Arrows: Turn Left / Right
        S / X: PTZ Tilt Up / Down
        A / D: PTZ Pan Left / Right
        Q / E: Zoom Out / In
    """
    def __init__(self, device="cuda:0", max_vel_speed=0.5, action_dim=4):
        self._device = device
        self.max_vel_speed = max_vel_speed
        if action_dim not in (2, 4, 5):
            raise ValueError(
                f"Inspection keyboard action_dim must be 2, 4, or 5; got {action_dim}"
            )
        self.action_dim = action_dim
        
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
        # Canonical action order. advance() returns only the prefix exposed by
        # the current environment (base-only, PT, or PTZ).
        self._base_command = np.zeros(5, dtype=np.float32)

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub') and self._keyboard_sub:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def advance(self) -> torch.Tensor:
        """Provides the current action tensor based on keyboard state.
        Shape is ``(1, action_dim)`` for one manually controlled environment.
        """
        command = np.zeros(5, dtype=np.float32)
        for key in self._pressed_keys:
            if key in self._INPUT_KEY_MAPPING:
                command += self._INPUT_KEY_MAPPING[key]
        return torch.tensor(
            [command[: self.action_dim]], dtype=torch.float32, device=self._device
        )

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
            "Q": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0]),  # Wide
            "E": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),   # Telephoto
        }
