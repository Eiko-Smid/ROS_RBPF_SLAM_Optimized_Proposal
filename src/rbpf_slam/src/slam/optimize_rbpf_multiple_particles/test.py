from dataclasses import fields
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import pandas as pd


class StepResultFlattener:
    """
    Converts one StepResult into one flat DataFrame row.

    Rules:
    - Fields ending in '_pose' are expanded into x, y and theta.
    - Scalar values are copied directly.
    - Other arrays/lists are skipped because they do not fit into one
      scalar DataFrame row.
    """

    POSE_COMPONENTS = ("x", "y", "theta")

    def __init__(self, result_type: Type[Any]) -> None:
        self.result_type = result_type

        # get_type_hints also works when
        # "from __future__ import annotations" is used.
        self.type_hints = get_type_hints(result_type)

    def flatten(self, step_result: Any) -> Dict[str, Any]:
        if not isinstance(step_result, self.result_type):
            raise TypeError(
                "Expected {}, got {}.".format(
                    self.result_type.__name__,
                    type(step_result).__name__,
                )
            )

        row: Dict[str, Any] = {}

        for field_info in fields(step_result):
            field_name = field_info.name
            value = getattr(step_result, field_name)
            annotation = self.type_hints.get(
                field_name,
                field_info.type,
            )

            # A 2D pose becomes three scalar columns.
            if self._is_pose_field(field_name):
                self._add_pose(
                    row=row,
                    field_name=field_name,
                    value=value,
                )
                continue

            # Do not place particle arrays or error arrays inside
            # individual DataFrame cells.
            if self._contains_ndarray(annotation):
                continue

            # Copy normal scalar values.
            if value is None or np.isscalar(value):
                row[field_name] = self._to_python_scalar(value)
                continue

            # Lists and other structured values are skipped as well.
            # They should be stored in a separate long-form DataFrame.

        return row

    def output_columns_for_field(self, field_name: str) -> List[str]:
        """
        Returns the DataFrame columns produced by a StepResult field.

        Example:
            'true_pose'
        becomes:
            ['true_pose_x', 'true_pose_y', 'true_pose_theta']
        """
        if self._is_pose_field(field_name):
            return [
                "{}_{}".format(field_name, component)
                for component in self.POSE_COMPONENTS
            ]

        return [field_name]

    @staticmethod
    def _is_pose_field(field_name: str) -> bool:
        return field_name.endswith("_pose")

    def _add_pose(
        self,
        row: Dict[str, Any],
        field_name: str,
        value: Any,
    ) -> None:
        output_columns = self.output_columns_for_field(field_name)

        if value is None:
            for column in output_columns:
                row[column] = None
            return

        pose = np.asarray(value, dtype=float).reshape(-1)

        if pose.size != 3:
            raise ValueError(
                "{} must contain exactly x, y and theta. "
                "Received shape {}.".format(
                    field_name,
                    np.asarray(value).shape,
                )
            )

        for column, pose_value in zip(output_columns, pose):
            row[column] = float(pose_value)

    @classmethod
    def _contains_ndarray(cls, annotation: Any) -> bool:
        if annotation is np.ndarray:
            return True

        origin = get_origin(annotation)

        if origin is Union:
            return any(
                cls._contains_ndarray(argument)
                for argument in get_args(annotation)
            )

        return False

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        # Convert np.float64, np.int64, np.bool_, etc. into
        # normal Python values.
        if isinstance(value, np.generic):
            return value.item()

        return value
    

class StepResultWriter:
    """
    Creates a DataFrame from StepResult objects and selects the fields
    that should be written.
    """

    def __init__(
        self,
        result_type: Type[Any],
        selected_fields: Optional[Sequence[str]] = None,
        excluded_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self.flattener = StepResultFlattener(result_type)

        self.selected_fields = (
            list(selected_fields)
            if selected_fields is not None
            else None
        )

        self.excluded_fields = (
            list(excluded_fields)
            if excluded_fields is not None
            else []
        )

    def to_dataframe(
        self,
        step_results: Iterable[Any],
    ) -> pd.DataFrame:
        rows = [
            self.flattener.flatten(step_result)
            for step_result in step_results
        ]

        df = pd.DataFrame.from_records(rows)

        if self.selected_fields is not None:
            selected_columns = self._resolve_fields(
                field_names=self.selected_fields,
                available_columns=set(df.columns),
                require_all=True,
            )

            df = df.loc[:, selected_columns]

        if self.excluded_fields:
            excluded_columns = self._resolve_fields(
                field_names=self.excluded_fields,
                available_columns=set(df.columns),
                require_all=False,
            )

            df = df.drop(columns=excluded_columns)

        return df

    def write_csv(
        self,
        step_results: Iterable[Any],
        file_path: str,
    ) -> pd.DataFrame:
        df = self.to_dataframe(step_results)
        df.to_csv(file_path, index=False)
        return df

    def _resolve_fields(
        self,
        field_names: Sequence[str],
        available_columns: Set[str],
        require_all: bool,
    ) -> List[str]:
        resolved_columns: List[str] = []
        missing_fields: List[str] = []

        for field_name in field_names:
            # Also allow the user to provide an already-expanded
            # DataFrame column such as "true_pose_x".
            if field_name in available_columns:
                resolved_columns.append(field_name)
                continue

            expanded_columns = (
                self.flattener.output_columns_for_field(field_name)
            )

            if all(
                column in available_columns
                for column in expanded_columns
            ):
                resolved_columns.extend(expanded_columns)
            elif require_all:
                missing_fields.append(field_name)

        if missing_fields:
            raise KeyError(
                "The following requested StepResult fields could not "
                "be converted into flat DataFrame columns: {}".format(
                    missing_fields
                )
            )

        # Remove duplicates while retaining order.
        return list(dict.fromkeys(resolved_columns))
    


def main():
    log_odds_increasing =  np.log(0.85 / 0.15)
    print(f"Log odds for increasing occupancy: {log_odds_increasing:.6f}")


if __name__ == "__main__":
    main()