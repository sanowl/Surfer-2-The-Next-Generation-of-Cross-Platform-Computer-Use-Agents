

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class OrchestratorAction(Enum):
    CREATE_PLAN = "create_plan"
    REPLAN = "replan"
    DELEGATE = "delegate"
    ANSWER = "answer"


@dataclass
class OrchestratorMemory:
    task: str
    plan: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_goal_index: int = 0
    observations: List[Any] = field(default_factory=list)

    def add_interaction(self, interaction: Dict[str, Any]):
        self.history.append(interaction)

    def update_plan(self, new_plan: List[Dict[str, Any]]):
        self.plan = new_plan

    def mark_goal_complete(self):
        if self.current_goal_index < len(self.plan):
            self.plan[self.current_goal_index]["status"] = "completed"
            self.current_goal_index += 1

    def get_current_goal(self) -> Optional[Dict[str, Any]]:
        if self.current_goal_index < len(self.plan):
            return self.plan[self.current_goal_index]
        return None


class Orchestrator:

    def __init__(
        self,
        model: Any,
        max_steps: int = 20,
        enable_adaptive_planning: bool = True
    ):
        self.model = model
        self.max_steps = max_steps
        self.enable_adaptive_planning = enable_adaptive_planning
        self.memory: Optional[OrchestratorMemory] = None
        self.current_step = 0

    def initialize(self, task: str) -> OrchestratorMemory:
        self.memory = OrchestratorMemory(task=task)
        self.current_step = 0
        return self.memory

    def should_use_orchestrator(self, task: str) -> bool:
        if not self.enable_adaptive_planning:
            return True

        complexity_indicators = [
            len(task.split()) > 20, 
            " and " in task.lower() or " then " in task.lower(), 
            "compare" in task.lower() or "find and" in task.lower(),  
            any(word in task.lower() for word in ["after", "before", "first", "second", "finally"]), 
        ]

        return sum(complexity_indicators) >= 2

    def create_plan(self, task: str, observations: List[Any]) -> List[Dict[str, Any]]:
        prompt = self._construct_planning_prompt(task, observations)

        response = self._call_model(prompt)
        plan = self._parse_plan(response)

        return plan

    def reflect_and_decide(
        self,
        navigator_report: Dict[str, Any],
        validator_feedback: Dict[str, Any]
    ) -> OrchestratorAction:
        self.memory.add_interaction({
            "navigator_report": navigator_report,
            "validator_feedback": validator_feedback,
            "step": self.current_step
        })
        if validator_feedback.get("success", False):
            self.memory.mark_goal_complete()
            if self.memory.current_goal_index >= len(self.memory.plan):
                return OrchestratorAction.ANSWER
            else:
                return OrchestratorAction.DELEGATE
        else:
            failure_count = self._count_recent_failures()

            if failure_count >= 2:
                # Multiple failures - replan
                return OrchestratorAction.REPLAN
            else:
                # Single failure - delegate again with feedback
                return OrchestratorAction.DELEGATE

    def replan(
        self,
        observations: List[Any],
        failure_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Construct replanning prompt
        prompt = self._construct_replanning_prompt(
            self.memory.task,
            self.memory.plan,
            self.memory.history,
            observations,
            failure_info
        )

        # Call LLM to generate new plan
        response = self._call_model(prompt)

        # Parse response into updated plan
        new_plan = self._parse_plan(response)

        # Update memory
        self.memory.update_plan(new_plan)

        return new_plan

    def delegate(self) -> Dict[str, Any]:
        current_goal = self.memory.get_current_goal()

        if current_goal is None:
            raise ValueError("No current goal to delegate")

        # Prepare subtask with context
        subtask = {
            "goal": current_goal["goal"],
            "verification_criteria": current_goal.get("verification_criteria", []),
            "context": {
                "overall_task": self.memory.task,
                "completed_goals": self.memory.plan[:self.memory.current_goal_index],
                "remaining_goals": self.memory.plan[self.memory.current_goal_index + 1:],
            }
        }

        return subtask

    def synthesize_answer(self, validated_results: List[Dict[str, Any]]) -> str:
        prompt = self._construct_synthesis_prompt(
            self.memory.task,
            validated_results
        )

        response = self._call_model(prompt)

        return response

    def step(
        self,
        navigator_report: Optional[Dict[str, Any]] = None,
        validator_feedback: Optional[Dict[str, Any]] = None,
        observations: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        self.current_step += 1

        if self.current_step > self.max_steps:
            return {
                "action": OrchestratorAction.ANSWER,
                "data": "Maximum steps reached",
                "terminated": True
            }

        # First step - create initial plan
        if self.current_step == 1:
            plan = self.create_plan(self.memory.task, observations or [])
            self.memory.update_plan(plan)
            return {
                "action": OrchestratorAction.DELEGATE,
                "data": self.delegate(),
                "terminated": False
            }

        # Subsequent steps - reflect and decide
        if navigator_report and validator_feedback:
            action = self.reflect_and_decide(navigator_report, validator_feedback)

            if action == OrchestratorAction.REPLAN:
                new_plan = self.replan(observations or [], validator_feedback)
                return {
                    "action": action,
                    "data": new_plan,
                    "terminated": False
                }
            elif action == OrchestratorAction.DELEGATE:
                return {
                    "action": action,
                    "data": self.delegate(),
                    "terminated": False
                }
            elif action == OrchestratorAction.ANSWER:
                # Collect all validated results
                validated_results = [
                    h.get("validator_feedback", {})
                    for h in self.memory.history
                    if h.get("validator_feedback", {}).get("success", False)
                ]
                answer = self.synthesize_answer(validated_results)
                return {
                    "action": action,
                    "data": answer,
                    "terminated": True
                }

        # Default: delegate current goal
        return {
            "action": OrchestratorAction.DELEGATE,
            "data": self.delegate(),
            "terminated": False
        }

    # Private helper methods

    def _call_model(self, prompt: str) -> str:
        # this needs an acc api key im adding a placeholder for now 
        if hasattr(self.model, 'generate'):
            return self.model.generate(prompt)
        elif callable(self.model):
            return self.model(prompt)
        else:
            raise NotImplementedError("Model must have 'generate' method or be callable")

    def _construct_planning_prompt(self, task: str, observations: List[Any]) -> str:
        """Construct prompt for initial planning"""
        return f"""You are a high-level task planner for a computer use agent.

Task: {task}

Current observations: {observations}

Break down this task into a sequence of verifiable subtasks. Each subtask should be:
1. Specific and actionable
2. Have clear verification criteria
3. Build logically on previous subtasks

Return a JSON list of subtasks in this format:
[
    {{
        "goal": "Description of what to achieve",
        "verification_criteria": ["criterion 1", "criterion 2"],
        "status": "pending"
    }},
    ...
]"""

    def _construct_replanning_prompt(
        self,
        task: str,
        current_plan: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        observations: List[Any],
        failure_info: Dict[str, Any]
    ) -> str:
        """Construct prompt for replanning after failure"""
        return f"""You are a high-level task planner for a computer use agent.

Original Task: {task}

Current Plan: {current_plan}

Execution History: {history}

Current Observations: {observations}

Failure Information: {failure_info}

The current plan has encountered failures. Analyze what went wrong and create an updated plan with a mitigation strategy.

Return a JSON list of subtasks in the same format as before."""

    def _construct_synthesis_prompt(
        self,
        task: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Construct prompt for synthesizing final answer"""
        return f"""You are synthesizing results from a multi-step task.

Original Task: {task}

Results from completed subtasks: {results}

Provide a clear, concise answer to the original task based on these results."""

    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured plan"""
        import json
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                plan = json.loads(json_str)
                return plan
            else:
                return [{
                    "goal": response.strip(),
                    "verification_criteria": [],
                    "status": "pending"
                }]
        except json.JSONDecodeError:
            return [{
                "goal": response.strip(),
                "verification_criteria": [],
                "status": "pending"
            }]

    def _count_recent_failures(self, window: int = 3) -> int:
        recent_history = self.memory.history[-window:]
        failures = sum(
            1 for h in recent_history
            if not h.get("validator_feedback", {}).get("success", False)
        )
        return failures
