import unittest, tempfile, importlib
from pathlib import Path
import numpy as np

class SmartTagTests(unittest.TestCase):
    def setUp(self):
        try:self.m=importlib.import_module('fcb_external.smart_tagging')
        except ModuleNotFoundError:self.fail('Smart Tagging module missing')
    def parse(self,body):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tags.xml';p.write_text('<file>'+body+'</file>')
            return self.m.load(p)
    def instance(self,start='0',end='4',labels=''):
        return '<instance><code>BUILD UP</code><start>'+start+'</start><end>'+end+'</end>'+labels+'</instance>'
    def label(self,group,value):return '<label><group>'+group+'</group><text>'+value+'</text></label>'
    def test_parse_rounding_unknown_and_conflicts(self):
        labels=self.label('Team','Red')+self.label('Team','Blue')+self.label('Half','1st Half')
        tag=self.parse(self.instance('.02','4',labels))[0]
        self.assertEqual(tag.start_frame,0);self.assertEqual(tag.end_frame,100)
        self.assertIsNone(tag.team);self.assertIn('team',tag.conflicts)
        self.assertIsNone(self.parse(self.instance(labels=self.label('Team','N/A')))[0].team)
        self.assertEqual(self.parse(''),[])
    def test_invalid_times_are_rejected(self):
        for a,b in [('nan','4'),('0','inf'),('4','3'),('-1','2'),('2','2')]:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):self.parse(self.instance(a,b))
    def test_optional_evidence_preserves_dead_probability_and_neutral_cases(self):
        labels=self.label('Team','Red')+self.label('Half','1st Half')
        tags=self.parse(self.instance(labels=labels));frames=np.arange(100);period=np.ones(100,int)
        side=self.m.owner_evidence(tags,frames,period,{'Red':0,'Blue':1})
        self.assertTrue(np.all(side[:25]==-1));self.assertTrue(np.all(side[25:75]==0));self.assertTrue(np.all(side[75:]==-1))
        p=np.tile([.3,.3,.4],(100,1));missing=np.ones(100,bool)
        q=self.m.fuse(p,side,missing)
        np.testing.assert_array_equal(q[:,2],p[:,2]);np.testing.assert_allclose(q.sum(1),1)
        self.assertGreater(q[50,0],p[50,0])
        np.testing.assert_array_equal(self.m.fuse(p,side,np.zeros(100,bool)),p)
        np.testing.assert_array_equal(self.m.owner_evidence(tags+tags,frames,period,{'Red':0}),side)
        other=self.parse(self.instance(labels=self.label('Team','Blue')+self.label('Half','1st Half')))
        conflict=self.m.owner_evidence(tags+other,frames,period,{'Red':0,'Blue':1})
        np.testing.assert_array_equal(self.m.fuse(p,conflict,missing),p)
        np.testing.assert_array_equal(self.m.owner_evidence(tags,frames,np.full(100,2),{'Red':0}),-1)
    def test_unsupported_phase_and_unknown_half_are_neutral(self):
        tag=self.parse(self.instance(labels=self.label('Team','Red')))
        np.testing.assert_array_equal(self.m.owner_evidence(tag,np.arange(100),np.ones(100),{'Red':0}),-1)
    def test_metadata_mapping_requires_explicit_distinct_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'meta.xml';p.write_text('<root><Team id="r" name="Red"/><Team id="b" name="Blue"/></root>')
            self.assertEqual(self.m.metadata_team_map(p),{'red':0,'blue':1})
            p.write_text('<root><Team id="r"/><Team id="b"/></root>')
            with self.assertRaises(ValueError):self.m.metadata_team_map(p)

    def test_integer_one_hot_probabilities_are_supported(self):
        p=np.array([[1,0,0],[0,1,0],[0,0,1]])
        np.testing.assert_array_equal(self.m.fuse(p,np.array([0,1,0]),np.ones(3,bool)),p)

    def test_wrong_tags_are_bounded_but_can_hurt(self):
        p=np.array([[.45,.40,.15],[.8,.1,.1]])
        q=self.m.fuse(p,np.ones(2,int),np.ones(2,bool))
        self.assertEqual(np.argmax(p[0]),0)
        self.assertEqual(np.argmax(q[0]),1)
        self.assertEqual(np.argmax(q[1]),0)
        np.testing.assert_allclose(q[:,1]/q[:,0],1.5*p[:,1]/p[:,0])
        np.testing.assert_array_equal(q[:,2],p[:,2])
        np.testing.assert_array_equal(p,np.array([[.45,.40,.15],[.8,.1,.1]]))

    def test_shifted_and_unsupported_tags_stay_local(self):
        labels=self.label('Team','Red')+self.label('Half','1st Half')
        tags=self.parse(self.instance('4','8',labels))
        frames=np.arange(250);period=np.ones(250,int)
        side=self.m.owner_evidence(tags,frames,period,{'Red':0})
        np.testing.assert_array_equal(np.flatnonzero(side==0),np.arange(125,175))
        from dataclasses import replace
        unsupported=[replace(tags[0],code='defensive block')]
        np.testing.assert_array_equal(self.m.owner_evidence(unsupported,frames,period,{'Red':0}),-1)
        np.testing.assert_array_equal(self.m.owner_evidence(tags,frames,period,{}),-1)
        p=np.tile([.3,.4,.3],(250,1))
        empty=self.m.owner_evidence([],frames,period,{'Red':0})
        np.testing.assert_array_equal(self.m.fuse(p,empty,np.ones(250,bool)),p)

    def test_audit_failure_is_operationally_visible_without_details(self):
        import contextlib, io, json
        from run_smart_tag_audit import run
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'report.json';stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                status=run(Path(tmp)/'absent-private-input',output)
            self.assertFalse(status)
            self.assertIn('needs_local_review',stream.getvalue())
            self.assertNotIn('absent-private-input',stream.getvalue())
            self.assertEqual(json.loads(output.read_text())['status'],'needs_local_review')

    def test_configured_adapter_and_disabled_path(self):
        from types import SimpleNamespace
        from fcb_external.smart_tagging import apply_candidate
        obs=SimpleNamespace(frame_id=np.arange(1,101),period=np.ones(100,int),ball_observed=np.zeros(100,bool))
        p=np.tile([.3,.3,.4],(100,1))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tags=root/'tags.xml';meta=root/'meta.xml'
            tags.write_text('<file>'+self.instance(labels=self.label('Team','Red')+self.label('Half','1st Half'))+'</file>')
            meta.write_text('<root><Team id="r" name="Red"/><Team id="b" name="Blue"/></root>')
            cfg={'phase_codes':['build up','progression'],'margin_frames':25,'odds_multiplier':1.5,'missing_ball_only':True,'preserve_dead_probability':True,'conflict_policy':'unchanged_v2'}
            q=apply_candidate(p,obs,tags,meta,cfg)
            self.assertGreater(q[50,0],p[50,0])
            np.testing.assert_array_equal(q[:,2],p[:,2])
            np.testing.assert_array_equal(apply_candidate(p,obs,None,None,None),p)
            cfg['missing_ball_only']=False
            with self.assertRaises(ValueError):apply_candidate(p,obs,tags,meta,cfg)
