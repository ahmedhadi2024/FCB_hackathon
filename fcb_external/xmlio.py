"""Streaming completeness and match-identity validation for external XML assets."""
from pathlib import Path
import xml.etree.ElementTree as ET


def validate_xml(path, match_id, required_tag):
    expected = 'DFL-MAT-'+match_id
    seen_id = False
    seen_tag = False
    stack = []
    with Path(path).open('rb') as handle:
        for event,elem in ET.iterparse(handle,events=('start','end')):
            if event == 'start':
                stack.append(elem)
                tag = elem.tag.rsplit('}',1)[-1]
                seen_tag |= tag == required_tag
                mid = elem.get('MatchId')
                if mid:
                    if mid != expected:
                        raise ValueError(f'Wrong match in external XML: expected {expected}')
                    seen_id = True
                if elem.get('TransmissionComplete','true').casefold() != 'true':
                    raise ValueError('Incomplete source transmission')
            else:
                stack.pop()
                if stack:
                    stack[-1].remove(elem)
                elem.clear()
    if not seen_id or not seen_tag:
        raise ValueError(f'External XML missing match identity or required {required_tag} structure')
