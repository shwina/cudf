from cudf._lib.nvtx import start_range, end_range
import time

i = start_range("my_range", color="green", domain="bomain")
time.sleep(5)
end_range(in)
