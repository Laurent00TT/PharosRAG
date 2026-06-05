def test_job_error_to_dict():
    from kb.jobs.errors import JobError

    err = JobError(error_type="ParseError", message="bad pdf", retryable=False)

    assert err.to_dict()["retryable"] is False
    assert err.to_dict()["error_type"] == "ParseError"


def test_job_status_constants_are_unique():
    from kb.jobs import models

    statuses = [
        models.JOB_QUEUED,
        models.JOB_RUNNING,
        models.JOB_SUCCEEDED,
        models.JOB_PARTIALLY_SUCCEEDED,
        models.JOB_FAILED,
        models.JOB_CANCELLING,
        models.JOB_CANCELLED,
    ]

    assert len(statuses) == len(set(statuses))
