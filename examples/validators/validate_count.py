"""Validate that an argument is an integer count greater than or equal to a minimum."""

from klygo import validators


# 1. Successful validation with default minimum=1
batch_size = 4
validators.validate_count(batch_size, "batch_size")
print("Valid batch size:", batch_size)

# 2. Successful validation with custom minimum=0
warmup_runs = 0
validators.validate_count(warmup_runs, "warmup_runs", minimum=0)
print("Valid warmup runs:", warmup_runs)

# 3. Reject booleans and non-integers with TypeError
try:
    validators.validate_count(True, "batch_size")
except TypeError as error:
    print("Caught expected TypeError:", error)

# 4. Reject values below minimum with ValueError
try:
    validators.validate_count(0, "batch_size", minimum=1)
except ValueError as error:
    print("Caught expected ValueError:", error)
