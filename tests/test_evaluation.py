from chemcrow_lite.evaluation import evaluate_run


def test_evaluate_run_detects_uncertainty_variants() -> None:
    task_spec = {"evaluation": {"expects_uncertainty": True}}
    result = evaluate_run(
        answer="The reaction outcome cannot be predicted with high confidence because the conditions are underspecified.",
        audit_path=None,
        task_spec=task_spec,
        run_mode="agent",
    )
    assert result["overall_pass"] is True
    assert result["checks"]["uncertainty"] is True
