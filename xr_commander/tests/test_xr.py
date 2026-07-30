from xr_commander.xr import wait_until_started


def test_the_wait_returns_once_the_server_reports_started():
    reports = iter([False, False, True])
    assert wait_until_started(lambda: next(reports), lambda: True, 1.0, tick_s=0.0)


def test_a_dead_server_thread_ends_the_wait_early():
    assert not wait_until_started(lambda: False, lambda: False, 30.0, tick_s=0.0)


def test_the_deadline_bounds_a_server_that_never_starts():
    assert not wait_until_started(lambda: False, lambda: True, 0.05, tick_s=0.0)
