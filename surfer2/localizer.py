

from typing import Tuple, Optional, Any, Dict, List
from dataclasses import dataclass
import json


@dataclass
class LocalizationResult:
    coordinates: Tuple[int, int]  # (x, y) pixel coordinates
    confidence: float
    element_description: str
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)


class Localizer:

    def __init__(
        self,
        model: Any,
        model_size: str = "7B",  # "7B" or "72B"
        default_confidence_threshold: float = 0.7
    ):
        self.model = model
        self.model_size = model_size
        self.confidence_threshold = default_confidence_threshold

    def localize(
        self,
        screenshot: Any,
        element_description: str,
        return_bounding_box: bool = False
    ) -> Tuple[int, int]:
        result = self._call_grounding_model(screenshot, element_description)

        if result.confidence < self.confidence_threshold:
            raise LocalizationError(
                f"Low confidence localization ({result.confidence:.2f}) for: {element_description}"
            )

        return result.coordinates

    def localize_multiple(
        self,
        screenshot: Any,
        element_descriptions: List[str]
    ) -> List[LocalizationResult]:
        results = []
        for desc in element_descriptions:
            try:
                coords = self.localize(screenshot, desc)
                results.append(LocalizationResult(
                    coordinates=coords,
                    confidence=1.0,  # Placeholder
                    element_description=desc
                ))
            except LocalizationError:
                # Skip elements that can't be localized
                continue

        return results

    def localize_with_fallback(
        self,
        screenshot: Any,
        primary_description: str,
        fallback_descriptions: List[str]
    ) -> Tuple[int, int]:
        # Try primary description
        try:
            return self.localize(screenshot, primary_description)
        except LocalizationError:
            pass
        for fallback in fallback_descriptions:
            try:
                return self.localize(screenshot, fallback)
            except LocalizationError:
                continue
        raise LocalizationError(
            f"Could not localize element with any description: "
            f"{[primary_description] + fallback_descriptions}"
        )

    def validate_localization(
        self,
        screenshot: Any,
        coordinates: Tuple[int, int],
        expected_description: str
    ) -> bool:
        # For now, return True (placeholder)
        return True

    def get_element_at_coordinates(
        self,
        screenshot: Any,
        coordinates: Tuple[int, int]
    ) -> str:
        return f"Element at ({coordinates[0]}, {coordinates[1]})"

    def _call_grounding_model(
        self,
        screenshot: Any,
        element_description: str
    ) -> LocalizationResult:
        if hasattr(self.model, 'ground'):
            result = self.model.ground(
                image=screenshot,
                query=element_description
            )
            return self._parse_holo_result(result, element_description)

        elif hasattr(self.model, 'localize'):
            result = self.model.localize(
                image=screenshot,
                description=element_description
            )
            return self._parse_generic_result(result, element_description)

        elif callable(self.model):
            result = self.model(screenshot, element_description)
            return self._parse_generic_result(result, element_description)

        else:
            raise NotImplementedError(
                "Model must have 'ground' or 'localize' method or be callable"
            )

    def _parse_holo_result(
        self,
        result: Any,
        element_description: str
    ) -> LocalizationResult:
        if isinstance(result, dict):
            x = result.get('x', 0)
            y = result.get('y', 0)
            confidence = result.get('confidence', 1.0)
            bbox = result.get('bbox')

            return LocalizationResult(
                coordinates=(x, y),
                confidence=confidence,
                element_description=element_description,
                bounding_box=bbox
            )
        elif isinstance(result, (tuple, list)) and len(result) >= 2:
            return LocalizationResult(
                coordinates=(int(result[0]), int(result[1])),
                confidence=1.0,
                element_description=element_description
            )
        else:
            raise LocalizationError(
                f"Unexpected result format from Holo1.5: {type(result)}"
            )

    def _parse_generic_result(
        self,
        result: Any,
        element_description: str
    ) -> LocalizationResult:
        if isinstance(result, dict):
            # Dictionary with coordinates
            if 'coordinates' in result:
                coords = result['coordinates']
                return LocalizationResult(
                    coordinates=(coords[0], coords[1]),
                    confidence=result.get('confidence', 1.0),
                    element_description=element_description
                )
            elif 'x' in result and 'y' in result:
                return LocalizationResult(
                    coordinates=(result['x'], result['y']),
                    confidence=result.get('confidence', 1.0),
                    element_description=element_description
                )

        elif isinstance(result, (tuple, list)) and len(result) >= 2:
            # Direct coordinate tuple
            return LocalizationResult(
                coordinates=(int(result[0]), int(result[1])),
                confidence=1.0,
                element_description=element_description
            )

        raise LocalizationError(
            f"Could not parse localization result: {result}"
        )


class LocalizationError(Exception):
    pass


class MockLocalizer(Localizer):

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        super().__init__(model=None, confidence_threshold=1.0)

    def _call_grounding_model(
        self,
        screenshot: Any,
        element_description: str
    ) -> LocalizationResult:
        return LocalizationResult(
            coordinates=(self.screen_width // 2, self.screen_height // 2),
            confidence=1.0,
            element_description=element_description
        )
