import numpy as np
import matplotlib.pyplot as plt

# Helper for precision curve, given total bits
def precision_curve(total_bits):
    e_range = np.arange(0, total_bits + 1)
    m_range = total_bits - e_range
    decimal_precision = m_range * np.log10(2)
    return e_range, decimal_precision

plt.figure(figsize=(10, 6))

# FP64 (52 mantissa + 11 exponent = 63 bits)
e64, dp64 = precision_curve(63)
plt.plot(e64, dp64, label='FP64 (52+11)', color='teal')

# FP32 (23 mantissa + 8 exponent = 31 bits)
e32, dp32 = precision_curve(31)
plt.plot(e32, dp32, label='FP32 (23+8)', color='royalblue')

# FP16 (10 mantissa + 5 exponent = 15 bits)
e16, dp16 = precision_curve(15)
plt.plot(e16, dp16, label='FP16 (10+5)', color='purple')

# FP8 (3 mantissa + 5 exponent = 8 bits, e.g. E4M3 or E5M2)
e8, dp8 = precision_curve(7)
plt.plot(e8, dp8, label='FP8 (3+5 or 4+4)', color='darkred')

# Optionally add a custom or emerging format, e.g. 4-bit (FP4)
e4, dp4 = precision_curve(3)
plt.plot(e4, dp4, label='FP4 (1+2)', color='orange')

# Labels and formatting
plt.title("Exponent Bits vs Decimal Precision for Common FP Formats")
plt.xlabel("Exponent Bits (e)")
plt.ylabel("Decimal Precision (log₁₀ digits)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
