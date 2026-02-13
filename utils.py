import struct

def create_wav_header(sample_rate: int, num_channels: int, bits_per_sample: int, data_size: int) -> bytes:
    header = b'RIFF'
    header += struct.pack('<I', 36 + data_size)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)
    header += struct.pack('<H', 1)
    header += struct.pack('<H', num_channels)
    header += struct.pack('<I', sample_rate)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    header += struct.pack('<I', byte_rate)
    header += struct.pack('<H', block_align)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    return header
