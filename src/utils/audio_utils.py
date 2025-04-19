_audio_file_cache = {}

def load_audio_file(path):
    from pydub import AudioSegment
    if path in _audio_file_cache:
        return _audio_file_cache[path]
    audio = AudioSegment.from_file(path)
    _audio_file_cache[path] = audio
    return audio
