import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from tests.fixtures import write_bundle


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def cli(self):
        try:
            return importlib.import_module('fcb_external.cli')
        except ModuleNotFoundError:
            self.fail('CLI not implemented')

    def config(self):
        meta,pos,events=write_bundle(self.root/'raw')
        config={'match_id':'TEST','metadata':str(meta),'positions':str(pos),
                'events':str(events),'output':str(self.root/'artifacts'),
                'source_revision':'synthetic','source_urls':[], 'coordinate_system':'dfl-centred-metres'}
        path=self.root/'config.json'
        path.write_text(json.dumps(config))
        return path

    def test_end_to_end_and_deterministic_repeat(self):
        cli=self.cli()
        cfg=self.config()
        obs=self.root/'artifacts'/'observations'
        targets=self.root/'artifacts'/'targets'
        pred=self.root/'predictions'
        reports=self.root/'reports'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['prepare','--config',str(cfg)]),0)
            self.assertEqual(cli.main(['baseline','--observations',str(obs),'--out',str(pred)]),0)
            self.assertEqual(cli.main(['evaluate','--observations',str(obs),'--targets',str(targets),
                                       '--predictions',str(pred),'--out',str(reports)]),0)
            initial={p.name:p.read_bytes() for p in reports.iterdir()}
            self.assertEqual(cli.main(['prepare','--config',str(cfg)]),0)
            self.assertEqual(cli.main(['baseline','--observations',str(obs),'--out',str(pred)]),0)
            self.assertEqual(cli.main(['evaluate','--observations',str(obs),'--targets',str(targets),
                                       '--predictions',str(pred),'--out',str(reports)]),0)
        self.assertEqual(initial,{p.name:p.read_bytes() for p in reports.iterdir()})
        report=json.loads((reports/'benchmark.json').read_text())
        self.assertEqual(len(report['variants']),4)
        self.assertEqual(report['development_match'],'TEST')

    def test_missing_input_fails_usefully(self):
        cfg=self.config()
        (self.root/'raw'/'positions.xml').unlink()
        err=io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.cli().main(['prepare','--config',str(cfg)]),1)
        self.assertIn('positions.xml',err.getvalue())

    def test_events_from_wrong_match_rejected(self):
        cfg=self.config()
        (self.root/'raw'/'events.xml').write_text(
            '<PutDataRequest><Event MatchId="DFL-MAT-WRONG"/></PutDataRequest>')
        err=io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.cli().main(['prepare','--config',str(cfg)]),1)
        self.assertIn('match',err.getvalue().lower())

    def test_predictions_linked_to_observation_content(self):
        cli=self.cli()
        cfg=self.config()
        obs=self.root/'artifacts'/'observations'
        pred=self.root/'predictions'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['prepare','--config',str(cfg)]),0)
            self.assertEqual(cli.main(['baseline','--observations',str(obs),'--out',str(pred)]),0)
        import numpy as np
        a=np.load(obs/'ball_xy_m.npy'); a[0]=[50,0]
        np.save(obs/'ball_xy_m.npy',a)
        err=io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(['evaluate','--observations',str(obs),
                '--targets',str(self.root/'artifacts'/'targets'),'--predictions',str(pred),
                '--out',str(self.root/'reports')]),1)
        self.assertIn('observation',err.getvalue().lower())


if __name__=='__main__':
    unittest.main()
