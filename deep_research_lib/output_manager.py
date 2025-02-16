# deep-research-python/deep_research_lib/output_manager.py
import sys
import os

class OutputManager:
    def __init__(self):
        self.progress_lines = 4
        self.progress_area = []
        self.initialized = False
        # Initialize terminal - only if it's a tty (interactive terminal)
        if sys.stdout.isatty():
            sys.stdout.write('\n' * self.progress_lines)
            self.initialized = True

    def log(self, *args):
        # Move cursor up to progress area
        if self.initialized:
            sys.stdout.write(f'\x1B[{self.progress_lines}A')
            # Clear progress area
            sys.stdout.write('\x1B[0J')
        # Print log message
        print(*args)
        # Redraw progress area if initialized
        if self.initialized:
            self.draw_progress()

    def update_progress(self, prog_data): # Renamed parameter to prog_data
        self.progress_area = [
            f"Depth:    [{self.get_progress_bar(prog_data['totalDepth'] - prog_data['currentDepth'], prog_data['totalDepth'])}] {round((prog_data['totalDepth'] - prog_data['currentDepth']) / prog_data['totalDepth'] * 100)}%", # Use prog_data here
            f"Breadth:  [{self.get_progress_bar(prog_data['totalBreadth'] - prog_data['currentBreadth'], prog_data['totalBreadth'])}] {round((prog_data['totalBreadth'] - prog_data['currentBreadth']) / prog_data['totalBreadth'] * 100)}%", # Use prog_data here
            f"Queries:  [{self.get_progress_bar(prog_data['completedQueries'], prog_data['totalQueries'])}] {round(prog_data['completedQueries'] / prog_data['totalQueries'] * 100)}%", # Use prog_data here
            f"Current:  {prog_data.get('currentQuery', '')}" # Use prog_data here
        ]
        self.draw_progress()

    def get_progress_bar(self, value, total):
        try:
            width = min(30, os.get_terminal_size().columns - 20) if sys.stdout.isatty() else 30 # Check if it's a tty before accessing terminal size
        except OSError: # Handle cases where terminal size can't be determined (e.g., not a tty)
            width = 30
        filled = round((width * value) / total)
        return '█' * filled + ' ' * (width - filled)

    def draw_progress(self):
        if not self.initialized or not self.progress_area:
            return

        # Move cursor to progress area
        try:
            terminal_height = os.get_terminal_size().rows if sys.stdout.isatty() else 24 # Check if tty before accessing terminal size
        except OSError:
            terminal_height = 24

        sys.stdout.write(f'\x1B[{terminal_height - self.progress_lines};1H')
        # Draw progress bars
        sys.stdout.write(chr(10).join(self.progress_area)) # use chr(10) for newline
        # Move cursor back to content area
        sys.stdout.write(f'\x1B[{terminal_height - self.progress_lines - 1};1H')