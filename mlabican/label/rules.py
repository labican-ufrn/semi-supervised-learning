import numpy as np

from mlabican.label.base import LabelingStrategy


class RuleBasedLabelingStrategy(LabelingStrategy):
    """
    FlexConC specific logic: Decides between Naive and Memory based on
    whether the predictions matched across iterations.
    """

    def label_instances(
        self, selected_indices: np.ndarray, **kwargs
    ) -> np.ndarray:
        pred_1_it = kwargs.get('pred_1_it', {})
        pred_x_it = kwargs.get('pred_x_it', {})
        cl_memory = kwargs.get('cl_memory', [])

        if (
            pred_1_it[selected_indices[0]]['classes']
            == pred_x_it[selected_indices[0]]['classes']
        ):
            return np.array(
                [pred_1_it[i]['classes'] for i in selected_indices]
            )

        return np.array([np.argmax(cl_memory[x]) for x in selected_indices])
