"""
Session-based logging system for DatabaseAgentEval.

All logs for a single run go into one timestamped session folder.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import suppress

with suppress(Exception):
    # Unix-only; used to make session numbering safe under parallel runs.
    import fcntl  # type: ignore


LOGS_ROOT = Path("logs")
SESSION_LOGS_DIR = LOGS_ROOT / "sessions"


class SessionLogger:
    """
    Centralized session-based logger that manages all logging for a run.
    
    All logs go into a single timestamped session folder.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not SessionLogger._initialized:
            self._setup_session()
            SessionLogger._initialized = True
    
    def _get_next_session_number(self) -> int:
        """Get the next sequential session number."""
        SESSION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Read or create session counter file.
        #
        # NOTE: This can be called from multiple processes when running evals in
        # parallel. Use a best-effort file lock on Unix to avoid collisions.
        counter_file = SESSION_LOGS_DIR / ".session_counter"
        try:
            # Use a+ so the file is created if missing.
            with open(counter_file, "a+") as f:
                # Acquire exclusive lock if available.
                flock = getattr(globals().get("fcntl", None), "flock", None)
                if flock is not None:
                    flock(f, globals()["fcntl"].LOCK_EX)

                f.seek(0)
                content = f.read().strip()
                try:
                    current_number = int(content) if content else 0
                except ValueError:
                    current_number = 0

                next_number = current_number + 1
                f.seek(0)
                f.truncate()
                f.write(str(next_number))
                f.flush()
                os.fsync(f.fileno())

                if flock is not None:
                    flock(f, globals()["fcntl"].LOCK_UN)
                return next_number
        except IOError:
            # As a last resort, fall back to a race-prone approach.
            current_number = 0
            if counter_file.exists():
                try:
                    current_number = int(counter_file.read_text().strip() or "0")
                except Exception:
                    current_number = 0
            next_number = current_number + 1
            counter_file.write_text(str(next_number))
            return next_number
    
    def _setup_session(self):
        """Initialize logging session with all logs in one folder."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Get next session number
        self.session_number = self._get_next_session_number()
        
        # Create session directory with both number and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = f"{self.session_number:04d}_{timestamp}"
        self.session_dir = SESSION_LOGS_DIR / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler - clean format
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        
        # Store file handlers
        self.file_handlers = {}
        
        # Create main log file
        # Attach to the root logger so all logs are captured.
        #
        # NOTE: DEBUG logs from SDKs can include full prompts/request bodies and
        # make main.log huge. Default eval logs to INFO; opt into DEBUG only when
        # investigating provider-level failures.
        # Examples:
        #   LOG_FILE_LEVEL=DEBUG python eval.py ...
        #   LOG_FILE_LEVEL=WARNING python eval.py ...
        file_level_name = (os.getenv("LOG_FILE_LEVEL") or "INFO").strip().upper()
        file_level = getattr(logging, file_level_name, logging.INFO)
        self.add_file_handler("", "main.log", level=file_level)
        
        # Reduce verbosity from noisy SDK/HTTP libraries. At DEBUG these can log
        # full request payloads, including the repeated scenario context.
        for noisy_logger in (
            "boto3",
            "botocore",
            "urllib3",
            "httpx",
            "httpcore",
            "openai",
            "openai._base_client",
        ):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        # Save session metadata
        metadata = {
            "session_number": self.session_number,
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "logs_directory": str(self.session_dir),
            # Eval summary fields (filled in by eval.py when available)
            "model": None,
            "timestamp": None,
            "total_scenarios": None,
            "correct": None,
            "incorrect": None,
            "errors": None,
            "accuracy": None,
        }
        with open(self.session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the given name."""
        return logging.getLogger(name)
    
    def add_file_handler(self, logger_name: str, filename: str = None, level: int = logging.DEBUG) -> logging.Logger:
        """
        Add a file handler to a specific logger - all go to session folder.
        
        Args:
            logger_name: Name of the logger
            filename: Log filename (will be in session folder)
            level: Logging level
        """
        logger = logging.getLogger(logger_name)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"{logger_name.replace('.', '_')}.log"
        
        # Always write to session directory
        filepath = self.session_dir / filename
        
        # Remove existing handler if any
        if logger_name in self.file_handlers:
            logger.removeHandler(self.file_handlers[logger_name])
            self.file_handlers[logger_name].close()
        
        # Create new file handler
        file_handler = logging.FileHandler(filepath)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        
        # Add handler
        logger.addHandler(file_handler)
        logger.setLevel(min(logger.level, level))
        
        # Store reference
        self.file_handlers[logger_name] = file_handler
        
        return logger
    
    def set_console_level(self, level: int):
        """Change console logging level."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            # FileHandler is a StreamHandler; exclude it so we don't unintentionally
            # change file logging levels.
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
    
    
class ExperimentLogger:
    """Logger for tracking experiments with structured data."""
    
    def __init__(self, experiment_id: str, session_dir: Path):
        self.experiment_id = experiment_id
        self.session_dir = session_dir
        
        # JSON lines file for structured data
        self.json_file = open(self.session_dir / "experiments.jsonl", "a")
        
        # Regular logger
        self.logger = logging.getLogger(f"experiment.{experiment_id}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a structured event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "experiment_id": self.experiment_id,
            **data
        }
        
        # Write to JSON
        self.json_file.write(json.dumps(entry) + "\n")
        self.json_file.flush()
        
        # Also log to console
        self.logger.info(f"{event_type}: {data.get('message', '')}")
    
    def log_scenario(self, scenario_name: str, stage: str, details: Dict[str, Any]):
        """Log scenario events."""
        self.log_event(
            "scenario",
            {
                "scenario": scenario_name,
                "stage": stage,
                **details
            }
        )
    
    def log_llm_call(self, model: str, request: str, response: str, tokens: Dict[str, int] = None):
        """Log LLM interactions."""
        self.log_event(
            "llm_call",
            {
                "model": model,
                "request_preview": request[:500],
                "response_preview": response[:500],
                "tokens": tokens
            }
        )
    
    def log_agent_decision(self, action: str, reasoning: str, sql: str = None):
        """Log agent decisions."""
        self.log_event(
            "agent_decision",
            {
                "action": action,
                "reasoning": reasoning,
                "sql": sql
            }
        )
    
    def close(self):
        """Close the experiment logger."""
        self.json_file.close()


# Global session logger instance
_session_logger = None


def get_session_logger() -> SessionLogger:
    """Get the global session logger."""
    global _session_logger
    if _session_logger is None:
        _session_logger = SessionLogger()
    return _session_logger


# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return get_session_logger().get_logger(name)


def add_file_logging(name: str, filename: str = None) -> logging.Logger:
    """Add file logging to a specific logger."""
    return get_session_logger().add_file_handler(name, filename)


def set_verbose(verbose: bool = True):
    """Set verbose mode (DEBUG level) for console."""
    level = logging.DEBUG if verbose else logging.INFO
    get_session_logger().set_console_level(level)


def create_experiment(experiment_id: str) -> ExperimentLogger:
    """Create an experiment logger."""
    session = get_session_logger()
    return ExperimentLogger(experiment_id, session.session_dir)


# Backward compatibility
class DatabaseAgentLogger:
    """Compatibility wrapper for existing code."""
    
    def __init__(self, base_dir: Path = None):
        self.session = get_session_logger()
        self.current_experiment = None
        self.current_experiment_id = None
        self._experiment_counter = 0
    
    def start_experiment(self, model_name: str, test_type: str, db_config: Dict[str, Any]) -> str:
        """Start an experiment."""
        exp_id = f"{model_name}_{test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_experiment = create_experiment(exp_id)
        self.current_experiment.log_event(
            "experiment_start",
            {
                "model": model_name,
                "test_type": test_type,
                "db_config": db_config
            }
        )
        # Set current experiment ID for backward compatibility
        self._experiment_counter += 1
        self.current_experiment_id = self._experiment_counter
        return exp_id
    
    def end_experiment(self, exp_id: int, summary: Dict[str, Any]):
        """End an experiment."""
        if self.current_experiment:
            self.current_experiment.log_event(
                "experiment_end",
                {
                    "experiment_id": exp_id,
                    "summary": summary
                }
            )
            self.current_experiment.close()
            self.current_experiment = None
            self.current_experiment_id = None
    
    def log_scenario_start(self, name: str, data: Dict[str, Any]):
        """Log scenario start."""
        if self.current_experiment:
            self.current_experiment.log_scenario(name, "start", data)
    
    def log_scenario_end(self, name: str, result: Dict[str, Any]):
        """Log scenario end."""
        if self.current_experiment:
            self.current_experiment.log_scenario(name, "end", result)
    
    def get_experiment_dir(self, exp_id: str) -> Path:
        """Get experiment directory - returns session directory for new structure."""
        return self.session.session_dir


# Create backward compatible instance
db_logger = DatabaseAgentLogger()


_app_logging_configured = False


def configure_app_logging(level: Optional[int] = None) -> None:
    """
    Configure lightweight logging for non-eval entrypoints (e.g., data viewer).

    This is intentionally less intrusive than SessionLogger: it never clears root
    handlers and will no-op if logging is already configured by the host (e.g.,
    uvicorn).
    """
    global _app_logging_configured
    if _app_logging_configured:
        return

    root_logger = logging.getLogger()
    if root_logger.handlers:
        _app_logging_configured = True
        return

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level or logging.INFO)

    root_logger.setLevel(level or logging.INFO)
    root_logger.addHandler(console_handler)
    _app_logging_configured = True


def get_app_logger(name: str) -> logging.Logger:
    """Get a logger for non-session apps (e.g., FastAPI server)."""
    configure_app_logging()
    return logging.getLogger(name)
