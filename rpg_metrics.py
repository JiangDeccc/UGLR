"""Small dependency-free metrics shared by RPG training and evaluation scripts."""


def macro_f1_score(targets, predictions):
    """Return unweighted mean of per-class F1 over labels present in y_true/y_pred."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    if not targets:
        return 0.0

    labels = sorted(set(targets) | set(predictions))
    per_class_f1 = []
    for label in labels:
        true_positive = sum(
            target == label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        false_positive = sum(
            target != label and prediction == label
            for target, prediction in zip(targets, predictions)
        )
        false_negative = sum(
            target == label and prediction != label
            for target, prediction in zip(targets, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(
            2 * true_positive / denominator if denominator > 0 else 0.0
        )

    return sum(per_class_f1) / len(per_class_f1)
