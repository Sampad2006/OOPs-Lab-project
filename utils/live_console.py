"""
Live Console — Mock terminal UI for real-time extraction feedback.

Renders a styled terminal-like log display inside Streamlit using
st.empty() containers. Supports multiple log levels with color coding
and timestamps.

Used by the Microscope view to replace the standard st.spinner with
a more informative, scientific-looking extraction log.
"""

import time
from datetime import datetime
from typing import Optional

import streamlit as st

import config


class LiveConsole:
    """
    A mock terminal UI that displays live log lines in Streamlit.

    Uses st.session_state to persist log lines across reruns,
    and an st.empty() container for flicker-free updates.

    Usage:
        console = LiveConsole(container=st.empty(), session_key="my_logs")
        console.log("INFO", "Starting extraction...")
        console.log("SUCCESS", "Extracted 400 words.")
        console.clear()
    """

    def __init__(
        self,
        container,
        session_key: str = "console_logs",
        max_lines: int = 50
    ):
        """
        Initialize the live console.

        Args:
            container:   A Streamlit container (from st.empty() or st.container()).
            session_key: Key in st.session_state to store log lines.
            max_lines:   Maximum number of log lines to keep (oldest are dropped).
        """
        self._container = container
        self._session_key = session_key
        self._max_lines = max_lines

        # Initialize session state buffer
        if self._session_key not in st.session_state:
            st.session_state[self._session_key] = []

    @property
    def lines(self):
        """Return the current log lines."""
        return st.session_state.get(self._session_key, [])

    def log(self, level: str, message: str, delay: float = 0.0) -> None:
        """
        Append a log line and re-render the console.

        Args:
            level:   Log level: INFO, SUCCESS, WARN, ERROR, DEBUG.
            message: The log message text.
            delay:   Optional delay in seconds after logging (for visual effect).
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color = config.LOG_COLORS.get(level.upper(), "#a0a0b8")

        log_entry = {
            "timestamp": timestamp,
            "level": level.upper(),
            "message": message,
            "color": color,
        }

        # Append and trim
        lines = st.session_state.get(self._session_key, [])
        lines.append(log_entry)
        if len(lines) > self._max_lines:
            lines = lines[-self._max_lines:]
        st.session_state[self._session_key] = lines

        # Re-render
        self._render()

        if delay > 0:
            time.sleep(delay)

    def info(self, message: str, delay: float = 0.0) -> None:
        """Shortcut for INFO level."""
        self.log("INFO", message, delay)

    def success(self, message: str, delay: float = 0.0) -> None:
        """Shortcut for SUCCESS level."""
        self.log("SUCCESS", message, delay)

    def warn(self, message: str, delay: float = 0.0) -> None:
        """Shortcut for WARN level."""
        self.log("WARN", message, delay)

    def error(self, message: str, delay: float = 0.0) -> None:
        """Shortcut for ERROR level."""
        self.log("ERROR", message, delay)

    def debug(self, message: str, delay: float = 0.0) -> None:
        """Shortcut for DEBUG level."""
        self.log("DEBUG", message, delay)

    def clear(self) -> None:
        """Clear all log lines."""
        st.session_state[self._session_key] = []
        self._render()

    def _render(self) -> None:
        """Render the console HTML into the container."""
        lines = st.session_state.get(self._session_key, [])
        log_html = self._build_html(lines)
        self._container.markdown(log_html, unsafe_allow_html=True)

    @staticmethod
    def _build_html(lines: list) -> str:
        """Build the full console HTML from log lines."""
        if not lines:
            return ""

        rows = []
        for entry in lines:
            level_badge = (
                f'<span style="color:{entry["color"]}; font-weight:600; '
                f'min-width:70px; display:inline-block;">'
                f'[{entry["level"]}]</span>'
            )
            rows.append(
                f'<div class="console-line">'
                f'<span style="color:#6a6a80;">{entry["timestamp"]}</span> '
                f'{level_badge} '
                f'<span style="color:#e8e8f0;">{entry["message"]}</span>'
                f'</div>'
            )

        lines_html = "\n".join(rows)

        return f"""
        <div class="live-console">
            <div class="console-header">
                <span class="console-dot red"></span>
                <span class="console-dot yellow"></span>
                <span class="console-dot green"></span>
                <span class="console-title">Extraction Console</span>
            </div>
            <div class="console-body" id="console-scroll">
                {lines_html}
            </div>
        </div>
        """
