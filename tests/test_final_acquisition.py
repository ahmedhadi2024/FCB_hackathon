import unittest
import download_final_evaluation as module


class FinalAcquisitionTests(unittest.TestCase):
    def test_asset_names_come_from_listing_instead_of_assumed_division(self):
        self.assertTrue(hasattr(module,'resolve_assets'),'Listed external asset resolution is missing')
        prefixes={'metadata':'02_01_matchinformation','events':'03_02_events_raw',
                  'positions':'04_03_positions_raw_observed'}
        listing={f'DFL_{p}_DFL-COM-000001_DFL-MAT-EXTERNAL.xml':{} for p in prefixes.values()}
        result=module.resolve_assets(listing,'EXTERNAL')
        self.assertEqual(set(result),set(prefixes))
        self.assertTrue(all('COM-000001' in name for name in result.values()))
        listing['DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-EXTERNAL.xml']={}
        with self.assertRaises(ValueError): module.resolve_assets(listing,'EXTERNAL')
