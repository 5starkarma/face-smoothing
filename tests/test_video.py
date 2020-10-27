import unittest
import shutil
import tempfile

from utils.video import delete_video


class TestVideoFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temp
        self.test_file = tempfile.TemporaryFile(prefix='TemporaryFile_', 
                                                suffix='.mp4')
    def test_delete_video(self):
        self.assertEqual(delete_video(self.test_file), None)

    def test_make_temp_dir(self):
        pass

    def test_split_video(self):
        pass

    def test_process_video(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
