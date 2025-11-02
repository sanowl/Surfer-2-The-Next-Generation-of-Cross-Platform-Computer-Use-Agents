

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class NavigatorAction(Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    PRESS_KEY = "press_key"
    DRAG = "drag"
    SWIPE = "swipe_element"  # For mobile
    ANSWER = "answer"  # Signal task completion
    WAIT = "wait"


@dataclass
class NavigatorMemory:
    task: str
    subtask: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[Any] = field(default_factory=list)

    def add_note(self, note: str):
        """Add a note extracted from observation"""
        self.notes.append(note)

    def add_step(self, thought: str, action: Dict[str, Any], observation: Any):
        """Add a step to the trajectory"""
        self.trajectory.append({
            "thought": thought,
            "action": action,
            "observation": observation
        })
        if observation is not None:
            self.screenshots.append(observation)

    def get_recent_context(self, k: int = 5) -> List[Dict[str, Any]]:
        """Get recent k steps from trajectory"""
        return self.trajectory[-k:]


class Navigator:

    def __init__(
        self,
        policy_model: Any,
        localizer: Any,
        max_steps: int = 50,
        environment: str = "web"  # "web", "desktop", "mobile"
    ):
        self.policy_model = policy_model
        self.localizer = localizer
        self.max_steps = max_steps
        self.environment = environment
        self.memory: Optional[NavigatorMemory] = None
        self.current_step = 0

    def initialize(self, task: str, subtask: Optional[Dict[str, Any]] = None):
        self.memory = NavigatorMemory(task=task, subtask=subtask)
        self.current_step = 0

    def step(self, screenshot: Any) -> Tuple[Dict[str, Any], bool]:
        self.current_step += 1

        if self.current_step > self.max_steps:
            return {
                "action": NavigatorAction.ANSWER,
                "data": "Maximum steps reached",
                "note": "",
                "thought": "Reached maximum steps, terminating"
            }, True
        note = self._extract_note(screenshot)
        self.memory.add_note(note)
        thought = self._generate_thought(screenshot, note)
        action = self._decide_action(screenshot, thought)
        grounded_action = self._ground_action(action, screenshot)
        self.memory.add_step(thought, grounded_action, screenshot)
        terminated = grounded_action["action"] == NavigatorAction.ANSWER.value

        return {
            "action": grounded_action["action"],
            "data": grounded_action.get("data", {}),
            "note": note,
            "thought": thought,
            "grounded_action": grounded_action
        }, terminated

    def execute(
        self,
        environment_executor: Any,
        initial_screenshot: Any,
        validator: Optional[Any] = None
    ) -> Dict[str, Any]:
        screenshot = initial_screenshot
        terminated = False

        while not terminated and self.current_step < self.max_steps:
            # Execute one step
            step_result, terminated = self.step(screenshot)

            if terminated:
                # Task completion signaled
                if validator is not None:
                    # Integrated validation before allowing termination
                    validation_result = validator.validate(
                        task=self.memory.task,
                        trajectory=self.memory.trajectory,
                        proposed_answer=step_result.get("data", ""),
                        screenshots=self.memory.screenshots[-5:]  # Last 5 screenshots
                    )

                    if not validation_result["success"]:
                        # Validation failed - resume execution with feedback
                        terminated = False
                        # Add validator feedback to context
                        self.memory.add_note(
                            f"Validator feedback: {validation_result['feedback']}"
                        )
                        # Continue loop
                        screenshot = environment_executor.get_screenshot()
                        continue

                # If no validator or validation passed, prepare final report
                return {
                    "success": True,
                    "answer": step_result.get("data", ""),
                    "trajectory": self.memory.trajectory,
                    "notes": self.memory.notes,
                    "steps": self.current_step
                }
            try:
                execution_result = environment_executor.execute_action(
                    step_result["grounded_action"]
                )
                screenshot = execution_result.get("screenshot")
            except Exception as e:
                # Action execution failed
                self.memory.add_note(f"Action execution failed: {str(e)}")
                screenshot = environment_executor.get_screenshot()

        return {
            "success": False,
            "answer": "Task incomplete - maximum steps reached",
            "trajectory": self.memory.trajectory,
            "notes": self.memory.notes,
            "steps": self.current_step
        }

    def _extract_note(self, screenshot: Any) -> str:
        prompt = self._construct_note_prompt(screenshot)
        response = self._call_policy_model(prompt, screenshot)

        note = self._parse_note(response)
        return note

    def _generate_thought(self, screenshot: Any, note: str) -> str:
        prompt = self._construct_thought_prompt(screenshot, note)
        response = self._call_policy_model(prompt, screenshot)
        thought = self._parse_thought(response)
        return thought

    def _decide_action(self, screenshot: Any, thought: str) -> Dict[str, Any]:
        prompt = self._construct_action_prompt(screenshot, thought)
        response = self._call_policy_model(prompt, screenshot)
        action = self._parse_action(response)
        return action

    def _ground_action(self, action: Dict[str, Any], screenshot: Any) -> Dict[str, Any]:
        action_type = action.get("action")
        localizable_actions = [
            NavigatorAction.CLICK.value,
            NavigatorAction.DRAG.value,
            NavigatorAction.SWIPE.value
        ]

        if action_type in localizable_actions:
            element_desc = action.get("element")

            if element_desc and not isinstance(element_desc, (tuple, list)):
                coordinates = self.localizer.localize(
                    screenshot=screenshot,
                    element_description=element_desc
                )
                grounded = action.copy()
                if action_type == NavigatorAction.CLICK.value:
                    grounded["x"] = coordinates[0]
                    grounded["y"] = coordinates[1]
                elif action_type == NavigatorAction.DRAG.value:
                    grounded["x_start"] = coordinates[0]
                    grounded["y_start"] = coordinates[1]
                elif action_type == NavigatorAction.SWIPE.value:
                    grounded["x_touch"] = coordinates[0]
                    grounded["y_touch"] = coordinates[1]

                return grounded

        return action

    def _call_policy_model(self, prompt: str, screenshot: Any) -> str:
        if hasattr(self.policy_model, 'generate'):
            return self.policy_model.generate(prompt, image=screenshot)
        elif callable(self.policy_model):
            return self.policy_model(prompt, screenshot)
        else:
            raise NotImplementedError("Policy model must have 'generate' method or be callable")

    def _construct_note_prompt(self, screenshot: Any) -> str:
        task_desc = self.memory.subtask.get("goal") if self.memory.subtask else self.memory.task

        return f"""You are a computer use agent extracting information from a screenshot.

Task: {task_desc}

Previous notes:
{chr(10).join(f"- {note}" for note in self.memory.notes[-3:])}

Look at the current screenshot and extract any relevant information that helps with the task.
Write a brief note about what you observe.

Note:"""

    def _construct_thought_prompt(self, screenshot: Any, note: str) -> str:
        """Construct prompt for thought generation"""
        task_desc = self.memory.subtask.get("goal") if self.memory.subtask else self.memory.task

        recent_trajectory = self.memory.get_recent_context(3)
        trajectory_str = "\n".join(
            f"Step {i+1}:\n  Thought: {step['thought']}\n  Action: {step['action']}"
            for i, step in enumerate(recent_trajectory)
        )

        return f"""You are a computer use agent reasoning about the next step.

Task: {task_desc}

Recent trajectory:
{trajectory_str}

Current note: {note}

Think step-by-step about what action to take next to make progress on the task.

Thought:"""

    def _construct_action_prompt(self, screenshot: Any, thought: str) -> str:
        """Construct prompt for action decision"""
        environment_actions = self._get_environment_actions()

        return f"""You are a computer use agent deciding on an action.

Thought: {thought}

Available actions: {environment_actions}

Decide on the specific action to take. Return a JSON object with the action details.

For click: {{"action": "click", "element": "description of element"}}
For type: {{"action": "type", "text": "text to type"}}
For scroll: {{"action": "scroll", "direction": "up/down", "amount": pixels}}
For answer: {{"action": "answer", "data": "final answer"}}

Action:"""

    def _get_environment_actions(self) -> List[str]:
        base_actions = ["click", "type", "scroll", "press_key", "wait", "answer"]

        if self.environment == "desktop":
            return base_actions + ["drag"]
        elif self.environment == "mobile":
            return base_actions + ["swipe_element", "drag"]
        else:  # web
            return base_actions

    def _parse_note(self, response: str) -> str:
        """Parse note from model response"""
        # Extract note portion
        if "Note:" in response:
            return response.split("Note:")[-1].strip()
        return response.strip()

    def _parse_thought(self, response: str) -> str:
        """Parse thought from model response"""
        if "Thought:" in response:
            return response.split("Thought:")[-1].strip()
        return response.strip()

    def _parse_action(self, response: str) -> Dict[str, Any]:
        try:
            # Try to extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                action = json.loads(json_str)
                return action
            else:
                return {"action": "wait", "duration": 1}
        except json.JSONDecodeError:
            return {"action": "wait", "duration": 1}
