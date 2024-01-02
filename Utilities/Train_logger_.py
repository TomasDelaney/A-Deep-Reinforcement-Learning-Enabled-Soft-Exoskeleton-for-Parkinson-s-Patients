import sys


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        pass  # You can implement this method if needed

    def close(self):
        self.logfile.close()


if __name__ == "__main__":
    # Create an instance of the Logger class
    logger = Logger("output.txt")

    # Redirect the standard output to the logger
    sys.stdout = logger

    # Print some information
    print("Hello, world!")
    print("This is some information.")

    # Restore the standard output and close the logger
    sys.stdout = logger.terminal
    logger.close()
