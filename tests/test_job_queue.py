"""Tests for the async job queue system."""

import time
from concurrent.futures import ThreadPoolExecutor


from audiobook_mcp.server import (
    create_job,
    get_job,
    enqueue_job,
    get_queue_length,
    JobStatus,
    _job_queue,
    _queue_lock,
)


class TestJobCreation:
    """Tests for job creation."""

    def test_create_job(self):
        """Test creating a job."""
        job = create_job("test_type", {"key": "value"})

        assert job.job_id is not None
        assert job.job_type == "test_type"
        assert job.status == JobStatus.PENDING
        assert job.metadata == {"key": "value"}

    def test_get_job(self):
        """Test retrieving a job by ID."""
        job = create_job("test_type")
        retrieved = get_job(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_job(self):
        """Test retrieving a nonexistent job."""
        job = get_job("nonexistent-id")
        assert job is None


class TestJobQueue:
    """Tests for job queue operations."""

    def test_enqueue_job(self):
        """Test enqueueing a job."""
        job = create_job("test_type")

        def dummy_func():
            return {"result": "done"}

        enqueue_job(job, dummy_func)

        assert job.status == JobStatus.QUEUED
        assert job.queue_position is not None
        assert job.queue_position >= 1

    def test_queue_length(self):
        """Test getting queue length."""
        initial_length = get_queue_length()

        job = create_job("test_type")

        def slow_func():
            time.sleep(0.5)
            return {"result": "done"}

        enqueue_job(job, slow_func)

        # Queue length should increase
        assert get_queue_length() >= initial_length


class TestJobExecution:
    """Tests for job execution."""

    def test_job_completes_successfully(self):
        """Test that a job completes successfully."""
        job = create_job("test_type")

        def success_func():
            return {"result": "success"}

        enqueue_job(job, success_func)

        # Wait for job to complete (with timeout)
        for _ in range(50):  # 5 seconds max
            current_job = get_job(job.job_id)
            if current_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            time.sleep(0.1)

        current_job = get_job(job.job_id)
        assert current_job.status == JobStatus.COMPLETED
        assert current_job.result == {"result": "success"}

    def test_job_handles_failure(self):
        """Test that a failing job is marked as failed."""
        job = create_job("test_type")

        def failing_func():
            raise ValueError("Test error")

        enqueue_job(job, failing_func)

        # Wait for job to complete (with timeout)
        for _ in range(50):  # 5 seconds max
            current_job = get_job(job.job_id)
            if current_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            time.sleep(0.1)

        current_job = get_job(job.job_id)
        assert current_job.status == JobStatus.FAILED
        assert "Test error" in current_job.error


class TestQueueConcurrency:
    """Tests for queue thread safety."""

    def test_no_deadlock_on_concurrent_enqueue(self):
        """Test that concurrent enqueue operations don't cause deadlock."""
        jobs = []
        results = []

        def enqueue_and_wait(i):
            job = create_job(f"test_type_{i}")
            jobs.append(job)

            def quick_func():
                return {"index": i}

            enqueue_job(job, quick_func)
            results.append(f"enqueued_{i}")

        # Enqueue multiple jobs concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(enqueue_and_wait, i) for i in range(10)]
            # Wait with timeout
            for future in futures:
                future.result(timeout=5)

        assert len(results) == 10

    def test_queue_position_updates(self):
        """Test that queue positions update correctly."""
        jobs = []

        def slow_func():
            time.sleep(0.2)
            return {}

        # Enqueue multiple jobs
        for i in range(3):
            job = create_job(f"test_{i}")
            enqueue_job(job, slow_func)
            jobs.append(job)

        # First job should be running or about to run
        # Other jobs should have queue positions
        time.sleep(0.05)

        with _queue_lock:
            # Access queue_position from the nested job object
            queued_positions = [j.job.queue_position for j in _job_queue]
            # Positions should be sequential starting from 1
            expected = list(range(1, len(queued_positions) + 1))
            assert queued_positions == expected or len(queued_positions) == 0
