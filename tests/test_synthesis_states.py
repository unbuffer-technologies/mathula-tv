from mathula_tv.models import JobManifest


def test_synthesis_state_transitions_are_strict():
    job=JobManifest("j","x","/x","a"*64,1,state="synthesis_queued")
    for state in ("synthesis_claimed","synthesizing","synthesis_ready"): job.transition(state)
    assert job.state=="synthesis_ready"
