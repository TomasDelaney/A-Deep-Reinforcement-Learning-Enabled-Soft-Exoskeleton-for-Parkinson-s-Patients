import time


def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    # Your script's code here
    # For example:
    for i in range(1000000):
        _ = i * 2

    time.sleep(7.15)

    # Record the end time
    end_time = time.time()

    # Calculate the total time taken
    execution_time = end_time - start_time

    # Convert to hours, minutes, and seconds
    hours, minutes, seconds = seconds_to_hms(int(execution_time))

    print(f"Script executed in {hours} hours, {minutes} minutes, and {seconds} seconds.")
