import pylsl

def select_inlet():
    streams = pylsl.resolve_streams()
    if not streams:
        raise RuntimeError("No LSL streams found. Ensure your EEG device is online.")

    print("\nAvailable LSL Streams:\n")
    for i, s in enumerate(streams):
        print(f"{i}: {s.name()} ({s.type()}) | srate={s.nominal_srate():.1f} | ch={s.channel_count()}")

    idx = int(input("\nSelect LSL stream index: "))
    chosen = streams[idx]
    inlet = pylsl.StreamInlet(chosen)

    info = inlet.info()
    print(f"\nConnected to '{chosen.name()}'")
    print(f"Nominal srate: {info.nominal_srate():.1f} Hz | Channels: {info.channel_count()}")
    return inlet
