"""Fictional DFL-shaped XML, never extracts from the competition match."""
from pathlib import Path

META = '''<PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-TEST" HomeTeamId="home" GuestTeamId="away"/>
<Environment PitchX="105" PitchY="68"/>
<Teams><Team TeamId="away" Role="guest"><Players><Player PersonId="a"/></Players></Team>
<Team TeamId="home" Role="home"><Players><Player PersonId="h"/>
<Player PersonId="sub"/></Players></Team></Teams>
</MatchInformation></PutDataRequest>'''


def frame(n, x=0, y=0, status='1', possession='1', t=None):
    if t is None:
        t = f'2023-01-01T00:00:{(n-1)//25:02d}.{((n-1)%25)*40:03d}+00:00'
    return (f'<Frame N="{n}" T="{t}" X="{x}" Y="{y}" '
            f'BallStatus="{status}" BallPossession="{possession}"/>')


def frameset(team, pid, rows, section='firstHalf'):
    return (f'<FrameSet TeamId="{team}" PersonId="{pid}" GameSection="{section}">'
            + ''.join(rows) + '</FrameSet>')


def write_bundle(root, sets=None, meta=META):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / 'metadata.xml'
    positions = root / 'positions.xml'
    events = root / 'events.xml'
    metadata.write_text(meta, encoding='utf-8')
    if sets is None:
        sets = [frameset('home', 'h', [frame(n, 0, 0) for n in range(1, 5)]),
                frameset('away', 'a', [frame(n, 10, 0) for n in range(1, 5)]),
                frameset('ball', 'ball', [frame(n, 0, 0, status='0' if n==4 else '1',
                                               possession='1' if n<3 else '2') for n in range(1,5)])]
    positions.write_text('<PutDataRequest><Positions MatchId="DFL-MAT-TEST">'
                         + ''.join(sets) + '</Positions></PutDataRequest>', encoding='utf-8')
    events.write_text('<PutDataRequest><Event MatchId="DFL-MAT-TEST"/></PutDataRequest>', encoding='utf-8')
    return metadata, positions, events
