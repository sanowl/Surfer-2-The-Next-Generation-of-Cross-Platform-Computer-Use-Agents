from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    success: bool
    feedback: str
    confidence: float = 0.0
    details: Optional[Dict[str, Any]] = None


class Validator:

    def __init__(
        self,
        judge_model: Any,
        k_screenshots: int = 5,
        enable_self_correction: bool = True
    ):
        self.judge_model = judge_model
        self.k_screenshots = k_screenshots
        self.enable_self_correction = enable_self_correction

    def validate(
        self,
        task: str,
        trajectory: List[Dict[str, Any]],
        proposed_answer: str,
        screenshots: List[Any],
        verification_criteria: Optional[List[str]] = None
    ) -> ValidationResult:
        recent_screenshots = screenshots[-self.k_screenshots:] if len(screenshots) > self.k_screenshots else screenshots
        prompt = self._construct_validation_prompt(
            task=task,
            trajectory=trajectory,
            proposed_answer=proposed_answer,
            verification_criteria=verification_criteria
        )
        response = self._call_judge_model(prompt, recent_screenshots)

        result = self._parse_validation_response(response)

        return result

    def validate_at_navigator_level(
        self,
        navigator_memory: Any,
        proposed_answer: str
    ) -> ValidationResult:
        return self.validate(
            task=navigator_memory.subtask.get("goal") if navigator_memory.subtask else navigator_memory.task,
            trajectory=navigator_memory.trajectory,
            proposed_answer=proposed_answer,
            screenshots=navigator_memory.screenshots,
            verification_criteria=navigator_memory.subtask.get("verification_criteria") if navigator_memory.subtask else None
        )

    def validate_at_orchestrator_level(
        self,
        task: str,
        navigator_report: Dict[str, Any],
        current_goal: Dict[str, Any]
    ) -> ValidationResult:
        return self.validate(
            task=current_goal.get("goal", task),
            trajectory=navigator_report.get("trajectory", []),
            proposed_answer=navigator_report.get("answer", ""),
            screenshots=navigator_report.get("screenshots", []),
            verification_criteria=current_goal.get("verification_criteria")
        )

    def multi_stage_verification(
        self,
        task: str,
        trajectory: List[Dict[str, Any]],
        proposed_answer: str,
        screenshots: List[Any],
        num_judges: int = 3
    ) -> ValidationResult:

        results = []

        for i in range(num_judges):
            # Use temperature > 0 for diversity
            result = self.validate(
                task=task,
                trajectory=trajectory,
                proposed_answer=proposed_answer,
                screenshots=screenshots
            )
            results.append(result)

        # Majority vote
        success_votes = sum(1 for r in results if r.success)
        majority_success = success_votes > (num_judges / 2)

        # Aggregate feedback
        if majority_success:
            # Find most detailed success feedback
            success_results = [r for r in results if r.success]
            best_result = max(success_results, key=lambda r: len(r.feedback))
            return ValidationResult(
                success=True,
                feedback=best_result.feedback,
                confidence=success_votes / num_judges,
                details={"votes": success_votes, "total": num_judges}
            )
        else:
            # Find most detailed failure feedback
            failure_results = [r for r in results if not r.success]
            best_result = max(failure_results, key=lambda r: len(r.feedback))
            return ValidationResult(
                success=False,
                feedback=best_result.feedback,
                confidence=(num_judges - success_votes) / num_judges,
                details={"votes": success_votes, "total": num_judges}
            )

    def _call_judge_model(self, prompt: str, screenshots: List[Any]) -> str:
        """Call the judge VLM model"""
        if hasattr(self.judge_model, 'generate'):
            # Multi-image input
            return self.judge_model.generate(prompt, images=screenshots)
        elif callable(self.judge_model):
            return self.judge_model(prompt, screenshots)
        else:
            raise NotImplementedError("Judge model must have 'generate' method or be callable")

    def _construct_validation_prompt(
        self,
        task: str,
        trajectory: List[Dict[str, Any]],
        proposed_answer: str,
        verification_criteria: Optional[List[str]] = None
    ) -> str:
        trajectory_str = ""
        for i, step in enumerate(trajectory[-10:]):  # Last 10 steps
            trajectory_str += f"\nStep {i+1}:\n"
            trajectory_str += f"  Thought: {step.get('thought', 'N/A')}\n"
            trajectory_str += f"  Action: {step.get('action', 'N/A')}\n"

        # Format criteria
        criteria_str = ""
        if verification_criteria:
            criteria_str = "\n\nVerification Criteria:\n"
            criteria_str += "\n".join(f"- {criterion}" for criterion in verification_criteria)

        prompt = f"""You are a validation judge for a computer use agent.

Task: {task}
{criteria_str}

Recent execution trajectory:
{trajectory_str}

Proposed Answer: {proposed_answer}

You have access to the most recent screenshots showing the final state.

Your job is to determine if the proposed answer correctly completes the task based on:
1. Observable evidence in the screenshots
2. The execution trajectory
3. The verification criteria (if provided)

Provide your assessment in this format:

SUCCESS: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONING: [Explain your decision based on observable evidence]
FEEDBACK: [If failed, what needs to be corrected? If succeeded, confirm what was achieved]

Assessment:"""

        return prompt

    def _parse_validation_response(self, response: str) -> ValidationResult:
        success = False
        confidence = 0.5
        feedback = response
        if "SUCCESS:" in response:
            success_line = [line for line in response.split('\n') if line.strip().startswith("SUCCESS:")][0]
            success = "YES" in success_line.upper()
        if "CONFIDENCE:" in response:
            try:
                confidence_line = [line for line in response.split('\n') if line.strip().startswith("CONFIDENCE:")][0]
                confidence_str = confidence_line.split("CONFIDENCE:")[-1].strip()
                confidence = float(confidence_str)
            except (ValueError, IndexError):
                confidence = 0.5

        # Parse FEEDBACK field
        if "FEEDBACK:" in response:
            feedback = response.split("FEEDBACK:")[-1].strip()
        elif "REASONING:" in response:
            feedback = response.split("REASONING:")[-1].strip()

        return ValidationResult(
            success=success,
            feedback=feedback,
            confidence=confidence
        )

    def self_correct(
        self,
        validation_result: ValidationResult,
        current_context: Dict[str, Any]
    ) -> str:
        if validation_result.success:
            return "Task completed successfully."

        prompt = f"""The previous attempt failed validation.

Validation Feedback: {validation_result.feedback}

Current Context: {current_context}

Based on the feedback, provide specific guidance on what needs to be corrected to successfully complete the task.

Correction Guidance:"""

        response = self._call_judge_model(prompt, [])

        return response
