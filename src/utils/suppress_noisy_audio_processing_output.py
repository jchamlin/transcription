from .logging_utils import enable_output_suppressions, suppress_stdout, suppress_warning, suppress_logging

def suppress_noisy_audio_processing_output():
    enable_output_suppressions(True)

    suppress_stdout("Model was trained with pyannote.audio")
    suppress_stdout("Model was trained with torch")
    suppress_stdout(">>Performing voice activity detection")

    suppress_logging("faster_whisper")
    suppress_logging("fsspec.local")
    suppress_logging("matplotlib")
    suppress_logging("numba")
    suppress_logging("pydub.converter")
    suppress_logging("pyannote")
    suppress_logging("pytorch_lightning")
    suppress_logging("pytorch_lightning.utilities.migration")
    suppress_logging("speechbrain.dataio.encoder")
    suppress_logging("speechbrain.utils.checkpoints")
    suppress_logging("speechbrain.utils.fetching")
    suppress_logging("speechbrain.utils.parameter_transfer")
    suppress_logging("torio._extension.utils")
    suppress_logging("urllib3.connectionpool")

    suppress_warning(category=FutureWarning)
    suppress_warning("Using SYMLINK strategy on Windows*")
    suppress_warning("Requested Pretrainer collection using symlinks*")
    suppress_warning(".*TensorFloat-32.*")
    suppress_warning(".*speechbrain\\.pretrained.*deprecated.*")
    suppress_warning(".*std\\(\\): degrees of freedom.*")

