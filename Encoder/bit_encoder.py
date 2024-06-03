class BitEncoder:
    def __init__(self, bits_per_token=8):
        self.bits_per_token = bits_per_token
        self.vocab_len = 2**self.bits_per_token
        
    def encode(self, x):
        return int(x, 2)