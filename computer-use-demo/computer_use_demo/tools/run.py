"""Utility to run shell commands asynchronously with a timeout."""

import asyncio
import logging

logger = logging.getLogger("computer_use_demo.tools.run")

TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


async def run(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
):
    """Run a shell command asynchronously with a timeout."""
    logger.info(f"Starting command execution: {cmd}")
    
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    
    logger.debug(f"Process started with PID: {process.pid}")

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        returncode = process.returncode or 0
        
        if returncode != 0:
            logger.warning(f"Command '{cmd}' returned non-zero exit code: {returncode}")
        else:
            logger.info(f"Command '{cmd}' completed successfully")
            
        return (
            returncode,
            maybe_truncate(stdout.decode(), truncate_after=truncate_after),
            maybe_truncate(stderr.decode(), truncate_after=truncate_after),
        )
    except asyncio.TimeoutError as exc:
        logger.error(f"Command '{cmd}' timed out after {timeout} seconds")
        try:
            process.kill()
            logger.info(f"Process {process.pid} killed after timeout")
        except ProcessLookupError:
            logger.warning(f"Process {process.pid} already terminated")
            pass
        raise TimeoutError(
            f"Command '{cmd}' timed out after {timeout} seconds"
        ) from exc
