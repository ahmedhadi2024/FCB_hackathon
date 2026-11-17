import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import download_idsse


class Response(io.BytesIO):
    headers = {}  # Server may omit Content-Length.


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_truncated_xml_not_promoted_without_content_length(self):
        name=download_idsse.FILES[0]
        with patch('urllib.request.urlopen',return_value=Response(b'<?xml version="1.0"?><PutDataRequest>')):
            with self.assertRaises(Exception):
                download_idsse.download(self.root)
        self.assertFalse((self.root/name).exists())

    def test_existing_wrong_match_is_not_called_verified(self):
        name=download_idsse.FILES[0]
        (self.root/name).write_text('<?xml version="1.0"?><PutDataRequest>'
            '<MatchInformation><General MatchId="DFL-MAT-WRONG"/></MatchInformation></PutDataRequest>')
        with patch('urllib.request.urlopen',side_effect=AssertionError('Unverified input was accepted')):
            with self.assertRaisesRegex(ValueError,'match'):
                download_idsse.download(self.root)
