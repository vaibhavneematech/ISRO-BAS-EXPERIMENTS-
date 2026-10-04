import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alerts.voice_alert import VoiceAlertSystem, AlertType
from src.protocol.state_machine import StateMachine, TransitionStatus

def verify_sound_events():
    events_played = []
    
    def on_play(path_str):
        name = pathlib.Path(path_str).name
        events_played.append(name)
        print(f"[SoundEvent] Fired audio: {name}")

    vas = VoiceAlertSystem(on_play_callback=on_play)

    # 1. Test clean positive beep for step complete
    print("\n--- Test 1: Successful Step Completion (CORRECT) ---")
    vas.play(AlertType.STEP_COMPLETE)
    assert "beep.wav" in events_played[-1] or "success_beep.wav" in events_played[-1]
    print("✓ Step Complete beep verified.")

    # 2. Test Skipped
    print("\n--- Test 2: Step Skipped ---")
    vas.play_if_new(AlertType.STEP_SKIPPED, "SKIPPED_STEP_3")
    assert "skipped" in events_played[-1].lower()
    print("✓ Skipped sound verified.")

    # 3. Test Wrong Sequence
    print("\n--- Test 3: Wrong Sequence ---")
    vas.play_if_new(AlertType.WRONG_SEQUENCE, "WRONG_ORDER_STEP_4")
    assert "wrongsequence" in events_played[-1].lower() or "wrogsequence" in events_played[-1].lower()
    print("✓ Wrong sequence sound verified.")

    # 4. Test Experiment Complete
    print("\n--- Test 4: All Steps Completed ---")
    vas.play(AlertType.EXPERIMENT_COMPLETE)
    assert "stepcomplete" in events_played[-1].lower() or "stepcompleted" in events_played[-1].lower()
    print("✓ Experiment Complete sound verified.")

    # 5. Verify single-event debounce (must not re-fire with same key)
    print("\n--- Test 5: Verify Debounce / Anti-Spam ---")
    before_len = len(events_played)
    vas.play_if_new(AlertType.STEP_SKIPPED, "SKIPPED_STEP_3")  # duplicate key
    assert len(events_played) == before_len, "Duplicate key should be suppressed!"
    print("✓ Duplicate sound suppression verified.")

    print("\nALL 4 SOUND CASES & DEBOUNCE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_sound_events()
