from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from .orchestrator import Orchestrator, OrchestratorAction
from .navigator import Navigator
from .validator import Validator, ValidationResult
from .localizer import Localizer


@dataclass
class Surfer2Config:
    use_orchestrator: bool = True
    orchestrator_model: Optional[Any] = None
    orchestrator_max_steps: int = 20

    navigator_policy_model: Any = None
    navigator_max_steps: int = 50

    validator_judge_model: Any = None
    validator_k_screenshots: int = 5

    localizer_model: Any = None
    localizer_size: str = "7B"

    environment: str = "web"

    benchmark: Optional[str] = None


class Surfer2:
    def __init__(self, config: Surfer2Config):
        self.config = config

        self.orchestrator = None
        if config.use_orchestrator and config.orchestrator_model:
            self.orchestrator = Orchestrator(
                model=config.orchestrator_model,
                max_steps=config.orchestrator_max_steps
            )

        self.navigator = Navigator(
            policy_model=config.navigator_policy_model,
            localizer=self._create_localizer(),
            max_steps=config.navigator_max_steps,
            environment=config.environment
        )

        self.validator = Validator(
            judge_model=config.validator_judge_model,
            k_screenshots=config.validator_k_screenshots
        )

        if config.benchmark:
            self._apply_benchmark_config(config.benchmark)

    def _create_localizer(self) -> Localizer:
        return Localizer(
            model=self.config.localizer_model,
            model_size=self.config.localizer_size
        )

    def _apply_benchmark_config(self, benchmark: str):
        benchmark = benchmark.lower()

        if benchmark == "osworld":
            self.config.use_orchestrator = False
            self.orchestrator = None
            self.navigator.max_steps = 100

        elif benchmark == "androidworld":
            self.config.use_orchestrator = False
            self.orchestrator = None
            self.navigator.max_steps = 150

    def execute(
        self,
        task: str,
        environment_executor: Any,
        initial_screenshot: Any
    ) -> Dict[str, Any]:
        if self.orchestrator and self.orchestrator.should_use_orchestrator(task):
            return self._execute_with_orchestrator(
                task, environment_executor, initial_screenshot
            )
        else:
            return self._execute_direct(
                task, environment_executor, initial_screenshot
            )

    def _execute_with_orchestrator(
        self,
        task: str,
        environment_executor: Any,
        initial_screenshot: Any
    ) -> Dict[str, Any]:
        self.orchestrator.initialize(task)

        orchestrator_result = self.orchestrator.step(
            observations=[initial_screenshot]
        )

        while not orchestrator_result.get("terminated", False):
            action = orchestrator_result["action"]

            if action == OrchestratorAction.DELEGATE:
                subtask = orchestrator_result["data"]

                self.navigator.initialize(task, subtask)

                navigator_report = self.navigator.execute(
                    environment_executor=environment_executor,
                    initial_screenshot=environment_executor.get_screenshot(),
                    validator=None
                )

                current_goal = self.orchestrator.memory.get_current_goal()
                validator_result = self.validator.validate_at_orchestrator_level(
                    task=task,
                    navigator_report=navigator_report,
                    current_goal=current_goal
                )

                orchestrator_result = self.orchestrator.step(
                    navigator_report=navigator_report,
                    validator_feedback=validator_result.__dict__,
                    observations=[environment_executor.get_screenshot()]
                )

            elif action == OrchestratorAction.REPLAN:
                orchestrator_result = self.orchestrator.step(
                    observations=[environment_executor.get_screenshot()]
                )

            elif action == OrchestratorAction.ANSWER:
                return {
                    "success": True,
                    "answer": orchestrator_result["data"],
                    "orchestrator_steps": self.orchestrator.current_step,
                    "plan": self.orchestrator.memory.plan,
                    "history": self.orchestrator.memory.history
                }

        return {
            "success": False,
            "answer": "Orchestrator terminated without answer",
            "orchestrator_steps": self.orchestrator.current_step
        }

    def _execute_direct(
        self,
        task: str,
        environment_executor: Any,
        initial_screenshot: Any
    ) -> Dict[str, Any]:
        self.navigator.initialize(task)

        result = self.navigator.execute(
            environment_executor=environment_executor,
            initial_screenshot=initial_screenshot,
            validator=self.validator
        )

        return result

    def execute_with_retries(
        self,
        task: str,
        environment_executor: Any,
        num_retries: int = 3
    ) -> Dict[str, Any]:
        best_result = None
        best_score = -1

        for attempt in range(num_retries):
            environment_executor.reset()
            initial_screenshot = environment_executor.get_screenshot()

            try:
                result = self.execute(
                    task=task,
                    environment_executor=environment_executor,
                    initial_screenshot=initial_screenshot
                )

                score = self._score_result(result)

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_result["attempt"] = attempt + 1

                if score >= 1.0:
                    return best_result

            except Exception as e:
                continue

        return best_result or {
            "success": False,
            "answer": "All attempts failed",
            "attempts": num_retries
        }

    def _score_result(self, result: Dict[str, Any]) -> float:
        if not result.get("success", False):
            return 0.0

        score = 1.0

        return score

    @classmethod
    def from_benchmark(
        cls,
        benchmark: str,
        models: Dict[str, Any]
    ) -> "Surfer2":
        benchmark = benchmark.lower()

        configs = {
            "webvoyager": Surfer2Config(
                use_orchestrator=True,
                orchestrator_model=models.get("o3"),
                orchestrator_max_steps=20,
                navigator_policy_model=models.get("claude_sonnet_4.5"),
                navigator_max_steps=50,
                validator_judge_model=models.get("gpt_4.1"),
                localizer_model=models.get("holo1.5_7b"),
                localizer_size="7B",
                environment="web",
                benchmark="webvoyager"
            ),
            "webarena": Surfer2Config(
                use_orchestrator=True,
                orchestrator_model=models.get("o3"),
                orchestrator_max_steps=20,
                navigator_policy_model=models.get("claude_sonnet_4.5"),
                navigator_max_steps=50,
                validator_judge_model=models.get("o3"),
                localizer_model=models.get("holo1.5_72b"),
                localizer_size="72B",
                environment="web",
                benchmark="webarena"
            ),
            "osworld": Surfer2Config(
                use_orchestrator=False,
                navigator_policy_model=models.get("claude_sonnet_4.5"),
                navigator_max_steps=100,
                validator_judge_model=models.get("o3"),
                localizer_model=models.get("holo1.5_72b"),
                localizer_size="72B",
                environment="desktop",
                benchmark="osworld"
            ),
            "androidworld": Surfer2Config(
                use_orchestrator=False,
                navigator_policy_model=models.get("o3"),
                navigator_max_steps=150,
                validator_judge_model=models.get("o3"),
                localizer_model=models.get("holo1.5_72b"),
                localizer_size="72B",
                environment="mobile",
                benchmark="androidworld"
            )
        }

        if benchmark not in configs:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        return cls(configs[benchmark])
