"""This file contains the event loop setup dependency."""

import asyncio
from asyncio import AbstractEventLoop

import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


LOOP = None


async def get_event_loop() -> AbstractEventLoop:
    """Get the event loop."""

    global LOOP
    if LOOP is None:
        LOOP = asyncio.get_event_loop()

    return LOOP
