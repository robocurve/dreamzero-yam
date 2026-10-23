"""CPU-only HTTP client compatibility; no Modal resources or network requests."""

import ssl
import unittest

import aiohttp
import certifi


class AiohttpCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_session_with_explicit_repaired_roots_closes(self):
        context = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=context),
            trust_env=False,
            timeout=aiohttp.ClientTimeout(total=2),
        ) as session:
            self.assertFalse(session.closed)
        self.assertTrue(session.closed)

    async def test_multipart_fixture_serializes_without_http(self):
        multipart = aiohttp.MultipartWriter("form-data")
        part = multipart.append(b"fixture-only", {"Content-Type": "application/octet-stream"})
        part.set_content_disposition("form-data", name="file", filename="fixture.bin")

        class Sink:
            def __init__(self):
                self.parts = []

            async def write(self, data):
                self.parts.append(data)

        sink = Sink()
        await multipart.write(sink)
        body = b"".join(sink.parts)
        self.assertIn(b"fixture-only", body)
        self.assertIn(b'filename="fixture.bin"', body)


if __name__ == "__main__":
    unittest.main()
