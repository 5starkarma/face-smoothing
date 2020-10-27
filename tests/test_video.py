import unittest
import shutil
import tempfile

from .utils import video

class TestVideoFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temp file
        self.test_file = tempfile.TemporaryFile(prefix='TemporaryFile_', 
                                                suffix='.mp4')
    def test_delete_video(self):
        file_del = video.delete_video(self.test_file)
        print(file_del, self.test_file)
        self.assertNotEqual(file_del, self.test_file)

    def test_make_temp_dir(self):
        pass

    def test_split_video(self):
        pass

    def test_process_video(self):
        pass

    def tearDown(self):
        pass
