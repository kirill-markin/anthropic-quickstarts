import asyncio
import os
import logging
from typing import Any, Literal

from .base import BaseAnthropicTool, CLIResult, ToolError, ToolResult

# Get logger
logger = logging.getLogger("computer_use_demo.tools.bash")


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False
        logger.debug("Bash session initialized")

    async def start(self):
        if self._started:
            logger.debug("Bash session already started")
            return

        logger.info("Starting new bash session")
        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.debug(f"Bash process started with PID: {self._process.pid}")

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            logger.warning("Attempted to stop a session that hasn't started")
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            logger.debug(
                f"Bash process already terminated with code {self._process.returncode}"
            )
            return
        logger.info(f"Terminating bash process with PID: {self._process.pid}")
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        logger.info(f"Running bash command: {command}")

        if not self._started:
            logger.error("Attempted to run command on session that hasn't started")
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            logger.warning(
                f"Bash has exited with returncode {self._process.returncode}"
            )
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            logger.error("Session timed out and must be restarted")
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        logger.debug(f"Sending command to bash: {command}")
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output: str = str(self._process.stdout._buffer.decode())  # pyright: ignore[reportAttributeAccessIssue] # type: ignore
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            logger.error(f"Command timed out after {self._timeout} seconds")
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error: str = str(self._process.stderr._buffer.decode())  # pyright: ignore[reportAttributeAccessIssue] # type: ignore
        if error.endswith("\n"):
            error = error[:-1]

        # log output and error
        if output:
            logger.debug(
                f"Command stdout: {output[:100]}{'...' if len(output) > 100 else ''}"
            )
        if error:
            logger.warning(
                f"Command stderr: {error[:100]}{'...' if len(error) > 100 else ''}"
            )

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue] # type: ignore
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue] # type: ignore

        return CLIResult(output=output, error=error)


class BashTool20250124(BaseAnthropicTool):
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    _session: _BashSession | None

    api_type: Literal["bash_20250124"] = "bash_20250124"
    name: Literal["bash"] = "bash"

    def __init__(self):
        self._session = None
        logger.debug("BashTool20250124 initialized")
        super().__init__()

    def to_params(self) -> Any:
        return {
            "type": self.api_type,
            "name": self.name,
        }

    async def __call__(
        self,
        command: str | None = None,
        restart: bool = False,
        **kwargs: dict[str, Any],
    ):
        logger.info(f"BashTool called with command={command}, restart={restart}")

        if restart:
            logger.info("Restarting bash session")
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            logger.info("First use - creating new bash session")
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        logger.error("Tool called with no command")
        raise ToolError("no command provided.")


class BashTool20241022(BashTool20250124):
    api_type: Literal["bash_20241022"] = "bash_20241022"  # pyright: ignore[reportIncompatibleVariableOverride]
